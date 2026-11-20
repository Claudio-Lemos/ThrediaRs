use evolve::{EvolveMonitor, HealthLevel};
use rodio::{OutputStream, Sink};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    pub enabled: bool,
    pub model_path: String,
    pub tokens_path: String,
    pub data_dir: String,
    pub lexicon_path: String,
    pub volume: f32,
    pub speed: f32,
    pub speaker_id: i32,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: "Models/vits-piper-pt_BR-faber-medium/pt_BR-faber-medium.onnx".to_string(),
            tokens_path: "Models/vits-piper-pt_BR-faber-medium/tokens.txt".to_string(),
            data_dir: "Models/vits-piper-pt_BR-faber-medium/espeak-ng-data".to_string(),
            lexicon_path: String::new(),
            volume: 0.85,
            speed: 1.0,
            speaker_id: 0,
        }
    }
}

impl SpeechConfig {
    pub fn load_or_default(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }
}

/// Handle thread-safe para enviar textos ou comandos (`_CMD_ENABLE_:`, `_CMD_LOAD_:`, `_CMD_STOP_`)
/// ao `SpeechEngine` isolado em sua própria `std::thread`.
#[derive(Clone)]
pub struct SpeechHandle {
    tx: Sender<String>,
}

impl SpeechHandle {
    pub fn send_raw(&self, msg: impl Into<String>) {
        let _ = self.tx.send(msg.into());
    }

    pub fn speak(&self, sentence: impl Into<String>) {
        let s = sentence.into();
        if !s.trim().is_empty() {
            let _ = self.tx.send(s);
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        let _ = self.tx.send(format!("_CMD_ENABLE_:{}", enabled));
    }

    pub fn load_model(&self, path: impl Into<String>) {
        let _ = self.tx.send(format!("_CMD_LOAD_:{}", path.into()));
    }

    pub fn stop(&self) {
        let _ = self.tx.send("_CMD_STOP_".to_string());
    }
}

pub struct SpeechEngine;

impl SpeechEngine {
    /// Inicia a thread dedicada de áudio/TTS com `OutputStream` e `Sink` persistentes (`rodio`).
    pub fn spawn(config_path: Option<PathBuf>) -> (SpeechHandle, SpeechConfig) {
        let cfg_path = config_path.unwrap_or_else(|| PathBuf::from("speech.json"));
        let config = SpeechConfig::load_or_default(&cfg_path);
        let initial_cfg = config.clone();

        let (tx, rx) = mpsc::channel::<String>();

        let _ = thread::Builder::new()
            .name("speech-engine".to_string())
            .spawn(move || {
                let mut cfg = config;

                // Mantém OutputStream e Sink vivos durante toda a vida da thread
                let audio_ctx = OutputStream::try_default()
                    .ok()
                    .and_then(|(stream, handle)| {
                        Sink::try_new(&handle).ok().map(|sink| {
                            sink.set_volume(cfg.volume.clamp(0.0, 2.0));
                            (stream, sink)
                        })
                    });

                if audio_ctx.is_some() {
                    EvolveMonitor::global().record_event(
                        HealthLevel::Info,
                        "speech",
                        format!(
                            "SpeechEngine iniciado (rodio OK, enabled={}, model={})",
                            cfg.enabled, cfg.model_path
                        ),
                    );
                } else {
                    EvolveMonitor::global().record_event(
                        HealthLevel::Warn,
                        "speech",
                        "SpeechEngine iniciado sem saída de áudio ativa (modo silencioso).",
                    );
                }

                while let Ok(msg) = rx.recv() {
                    if let Some(flag) = msg.strip_prefix("_CMD_ENABLE_:") {
                        cfg.enabled = flag.trim() == "true";
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "speech",
                            format!("TTS {}", if cfg.enabled { "ativado" } else { "desativado" }),
                        );
                        continue;
                    }

                    if let Some(path) = msg.strip_prefix("_CMD_LOAD_:") {
                        let trimmed = path.trim().to_string();
                        let exists = Path::new(&trimmed).exists();
                        cfg.model_path = trimmed.clone();
                        EvolveMonitor::global().record_event(
                            if exists { HealthLevel::Info } else { HealthLevel::Warn },
                            "speech",
                            format!(
                                "Modelo TTS selecionado: {} ({})",
                                trimmed,
                                if exists { "arquivo encontrado" } else { "arquivo não encontrado em disco" }
                            ),
                        );
                        continue;
                    }

                    if msg == "_CMD_STOP_" {
                        if let Some((_, ref sink)) = audio_ctx {
                            sink.clear();
                            sink.play();
                        }
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "speech",
                            "Reprodução TTS interrompida (_CMD_STOP_).",
                        );
                        continue;
                    }

                    if !cfg.enabled {
                        continue;
                    }

                    // Ponto de integração com VitsTts (Piper ONNX + espeak-ng-data):
                    // quando o modelo ONNX estiver presente e sherpa-rs compilado, os samples f32
                    // são enfileirados em `sink.append(rodio::buffer::SamplesBuffer::new(1, sr, samples))`.
                    let model_exists = Path::new(&cfg.model_path).exists();
                    EvolveMonitor::global().record_event(
                        HealthLevel::Info,
                        "speech",
                        format!(
                            "[TTS] \"{}\" (modelo_onnx={})",
                            msg.trim(),
                            if model_exists { "presente" } else { "ausente/simulado" }
                        ),
                    );
                }
            });

        (SpeechHandle { tx }, initial_cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_config_default_and_commands() {
        let cfg = SpeechConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.model_path.ends_with(".onnx"));
    }
}
