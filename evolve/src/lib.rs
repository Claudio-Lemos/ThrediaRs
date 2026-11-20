use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use sysinfo::{Components, Pid, System};
use tracing::{error, info, warn};

const MAX_EVENTS: usize = 512;
const MAX_SAMPLES: usize = 60;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub cpu_percent: f32,
    pub process_ram_mb: u64,
    pub last_llm_latency_ms: f64,
    pub whisper_accuracy_pct: f32,
    pub ml_action_success_pct: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthLevel {
    Info,
    Warn,
    Error,
    Fatal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthEvent {
    pub timestamp: DateTime<Utc>,
    pub level: HealthLevel,
    pub module: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSnapshot {
    pub cpu_usage_percent: f32,
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub process_memory_mb: u64,
    pub max_temp_celsius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfStats {
    pub operation: String,
    pub count: u64,
    pub avg_ms: f64,
    pub max_ms: f64,
    pub last_ms: f64,
}

#[derive(Debug, Default)]
struct PerfAccum {
    count: u64,
    total_ms: f64,
    max_ms: f64,
    last_ms: f64,
}

pub struct EvolveMonitor {
    system: Mutex<System>,
    components: Mutex<Components>,
    events: RwLock<VecDeque<HealthEvent>>,
    samples: RwLock<VecDeque<TelemetrySample>>,
    perf: RwLock<HashMap<String, PerfAccum>>,
    diag_path: RwLock<PathBuf>,
    whisper_accuracy_pct: RwLock<f32>,
    ml_actions: RwLock<(u64, u64)>, // (success, total)
}

static GLOBAL_MONITOR: Lazy<EvolveMonitor> = Lazy::new(EvolveMonitor::new);

impl EvolveMonitor {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let components = Components::new_with_refreshed_list();
        Self {
            system: Mutex::new(system),
            components: Mutex::new(components),
            events: RwLock::new(VecDeque::with_capacity(MAX_EVENTS)),
            samples: RwLock::new(VecDeque::with_capacity(MAX_SAMPLES)),
            perf: RwLock::new(HashMap::new()),
            diag_path: RwLock::new(PathBuf::from("Diag.json")),
            whisper_accuracy_pct: RwLock::new(100.0),
            ml_actions: RwLock::new((0, 0)),
        }
    }

    pub fn global() -> &'static Self {
        &GLOBAL_MONITOR
    }

    pub fn set_diag_path(&self, path: impl Into<PathBuf>) {
        *self.diag_path.write() = path.into();
    }

    pub fn init_tracing() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .try_init();
    }

    pub fn snapshot(&self) -> SystemSnapshot {
        let mut sys = self.system.lock();
        sys.refresh_cpu_usage();
        sys.refresh_memory();

        let pid = Pid::from_u32(std::process::id());
        sys.refresh_process(pid);

        let cpu_usage_percent = sys.global_cpu_info().cpu_usage();
        let total_memory_mb = sys.total_memory() / (1024 * 1024);
        let used_memory_mb = sys.used_memory() / (1024 * 1024);
        let process_memory_mb = sys
            .process(pid)
            .map(|p| p.memory() / (1024 * 1024))
            .unwrap_or(0);

        let mut comps = self.components.lock();
        comps.refresh_list();
        comps.refresh();
        let max_temp_celsius = comps
            .iter()
            .map(|c| c.temperature())
            .filter(|t| t.is_finite() && *t > 0.0)
            .fold(0.0_f32, f32::max);

        SystemSnapshot {
            cpu_usage_percent,
            total_memory_mb,
            used_memory_mb,
            process_memory_mb,
            max_temp_celsius,
        }
    }

    pub fn record_event(
        &self,
        level: HealthLevel,
        module: impl Into<String>,
        message: impl Into<String>,
    ) {
        let module = module.into();
        let message = message.into();

        match level {
            HealthLevel::Info => info!(target: "evolve", "[{}] {}", module, message),
            HealthLevel::Warn => warn!(target: "evolve", "[{}] {}", module, message),
            HealthLevel::Error => error!(target: "evolve", "[{}] {}", module, message),
            HealthLevel::Fatal => error!(target: "evolve", "[FATAL][{}] {}", module, message),
        }

        let event = HealthEvent {
            timestamp: Utc::now(),
            level,
            module,
            message,
        };

        {
            let mut events = self.events.write();
            if events.len() >= MAX_EVENTS {
                events.pop_front();
            }
            events.push_back(event);
        }

        if matches!(level, HealthLevel::Error | HealthLevel::Fatal) {
            let _ = self.export_diag_json();
        }
    }

    pub fn recent_events(&self) -> Vec<HealthEvent> {
        self.events.read().iter().cloned().collect()
    }

    pub fn record_latency(&self, operation: &str, elapsed_ms: f64) {
        let mut perf = self.perf.write();
        let entry = perf.entry(operation.to_string()).or_default();
        entry.count += 1;
        entry.total_ms += elapsed_ms;
        entry.last_ms = elapsed_ms;
        if elapsed_ms > entry.max_ms {
            entry.max_ms = elapsed_ms;
        }
    }

    pub fn get_performance_summary(&self) -> Vec<PerfStats> {
        let perf = self.perf.read();
        let mut out: Vec<PerfStats> = perf
            .iter()
            .map(|(op, acc)| PerfStats {
                operation: op.clone(),
                count: acc.count,
                avg_ms: if acc.count > 0 {
                    acc.total_ms / acc.count as f64
                } else {
                    0.0
                },
                max_ms: acc.max_ms,
                last_ms: acc.last_ms,
            })
            .collect();
        out.sort_by(|a, b| a.operation.cmp(&b.operation));
        out
    }

    pub fn log_health(&self) -> SystemSnapshot {
        let snap = self.snapshot();
        let last_llm_latency_ms = self
            .perf
            .read()
            .get("brain_generate")
            .map(|p| p.last_ms)
            .unwrap_or(0.0);
        let sample = TelemetrySample {
            cpu_percent: snap.cpu_usage_percent,
            process_ram_mb: snap.process_memory_mb,
            last_llm_latency_ms,
            whisper_accuracy_pct: self.whisper_accuracy(),
            ml_action_success_pct: self.ml_success_rate(),
        };
        {
            let mut samples = self.samples.write();
            if samples.len() >= MAX_SAMPLES {
                samples.pop_front();
            }
            samples.push_back(sample);
        }

        let msg = format!(
            "[HEALTH] CPU: {:.1}% | RAM: {}/{} MB (Proc: {} MB) | Temp: {:.1}°C",
            snap.cpu_usage_percent,
            snap.used_memory_mb,
            snap.total_memory_mb,
            snap.process_memory_mb,
            snap.max_temp_celsius
        );
        self.record_event(HealthLevel::Info, "evolve", msg);
        snap
    }

    pub fn recent_samples(&self) -> Vec<TelemetrySample> {
        self.samples.read().iter().cloned().collect()
    }

    pub fn set_whisper_accuracy(&self, pct: f32) {
        *self.whisper_accuracy_pct.write() = pct.clamp(0.0, 100.0);
    }

    pub fn whisper_accuracy(&self) -> f32 {
        *self.whisper_accuracy_pct.read()
    }

    pub fn record_ml_action(&self, success: bool) {
        let mut counts = self.ml_actions.write();
        counts.1 += 1;
        if success {
            counts.0 += 1;
        }
    }

    pub fn ml_success_rate(&self) -> f32 {
        let (ok, total) = *self.ml_actions.read();
        if total == 0 {
            100.0
        } else {
            (ok as f32 / total as f32) * 100.0
        }
    }

    pub fn export_diag_json(&self) -> std::io::Result<()> {
        let diag_events: Vec<HealthEvent> = self
            .events
            .read()
            .iter()
            .filter(|e| matches!(e.level, HealthLevel::Error | HealthLevel::Fatal))
            .cloned()
            .collect();

        let payload = serde_json::json!({
            "exported_at": Utc::now(),
            "error_count": diag_events.len(),
            "whisper_accuracy_pct": self.whisper_accuracy(),
            "ml_action_success_pct": self.ml_success_rate(),
            "performance": self.get_performance_summary(),
            "events": diag_events,
        });

        let path = self.diag_path.read().clone();
        let json = serde_json::to_string_pretty(&payload).map_err(std::io::Error::other)?;
        fs::write(path, json)
    }

    /// Lê `Diag.json` (se existir e tiver erros) para a rotina "The Blind Vault" / Modo Sonho.
    pub fn read_diag_errors(&self) -> Option<String> {
        let path = self.diag_path.read().clone();
        let raw = fs::read_to_string(&path).ok()?;
        let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let count = val.get("error_count").and_then(|v| v.as_u64()).unwrap_or(0);
        if count == 0 {
            return None;
        }
        let events = val.get("events").and_then(|v| v.as_array())?;
        let mut summary = format!("Erros registrados em {:?} ({} total):\n", path, count);
        for ev in events.iter().take(10) {
            let module = ev.get("module").and_then(|v| v.as_str()).unwrap_or("?");
            let msg = ev.get("message").and_then(|v| v.as_str()).unwrap_or("");
            summary.push_str(&format!("- [{}] {}\n", module, msg));
        }
        Some(summary)
    }

    /// Grava o relatório isolado do Modo Sonho (`DreamReport.md`) ao lado de `Diag.json`.
    pub fn write_dream_report(&self, report: &str) -> std::io::Result<PathBuf> {
        let diag_path = self.diag_path.read().clone();
        let report_path = diag_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.join("DreamReport.md"))
            .unwrap_or_else(|| PathBuf::from("DreamReport.md"));
        let content = format!(
            "# Relatório de Diagnóstico — Modo Sonho (Blind Vault)\n\nGerado em: {}\n\n{}\n",
            Utc::now().to_rfc3339(),
            report.trim()
        );
        fs::write(&report_path, content)?;
        Ok(report_path)
    }
}

impl Default for EvolveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard que mede latência de uma operação em milissegundos e registra no EvolveMonitor.
pub struct PerfTimer {
    operation: String,
    start: Instant,
}

impl PerfTimer {
    pub fn start(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        let ms = self.elapsed_ms();
        EvolveMonitor::global().record_latency(&self.operation, ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_timer_and_summary() {
        let monitor = EvolveMonitor::new();
        monitor.record_latency("llm_infer", 42.5);
        monitor.record_latency("llm_infer", 57.5);

        let summary = monitor.get_performance_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].operation, "llm_infer");
        assert_eq!(summary[0].count, 2);
        assert!((summary[0].avg_ms - 50.0).abs() < 1e-6);
        assert!((summary[0].max_ms - 57.5).abs() < 1e-6);
    }

    #[test]
    fn test_ring_buffer_and_diag_export() {
        let monitor = EvolveMonitor::new();
        let tmp = std::env::temp_dir().join("thrediars_test_diag.json");
        monitor.set_diag_path(&tmp);

        monitor.record_event(HealthLevel::Info, "test", "ok");
        monitor.record_event(HealthLevel::Error, "test", "simulated failure");

        let events = monitor.recent_events();
        assert_eq!(events.len(), 2);
        assert!(tmp.exists());

        let content = fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("simulated failure"));
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_snapshot_runs() {
        let monitor = EvolveMonitor::new();
        let snap = monitor.log_health();
        assert!(snap.total_memory_mb > 0);
        assert_eq!(monitor.recent_samples().len(), 1);
    }

    #[test]
    fn test_blind_vault_and_ml_metrics() {
        let monitor = EvolveMonitor::new();
        let tmp = std::env::temp_dir().join("thrediars_blind_vault_diag.json");
        monitor.set_diag_path(&tmp);

        monitor.set_whisper_accuracy(94.5);
        monitor.record_ml_action(true);
        monitor.record_ml_action(false);
        assert!((monitor.whisper_accuracy() - 94.5).abs() < 1e-4);
        assert!((monitor.ml_success_rate() - 50.0).abs() < 1e-4);

        monitor.record_event(HealthLevel::Error, "speech", "Audio device busy");
        let summary = monitor.read_diag_errors().unwrap();
        assert!(summary.contains("Audio device busy"));

        let report_path = monitor.write_dream_report("O dispositivo de áudio estava ocupado.").unwrap();
        assert!(report_path.exists());

        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(&report_path);
    }
}

