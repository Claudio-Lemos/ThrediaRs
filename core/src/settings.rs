use brain::{ComputeBackend, CustomPromptPreset, SamplerConfig, SharedState};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Todas as configurações editáveis pela GUI, serializadas em `settings.json`
/// (ou exportadas/importadas para qualquer caminho JSON escolhido pelo usuário).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub llm_model_path: String,
    pub slm_model_path: String,
    pub compute_backend: ComputeBackend,
    pub move_to_modelos: bool,
    pub llm_temp: f32,
    pub llm_ctx_size: u32,
    pub sampler: SamplerConfig,
    pub show_thinking: bool,
    pub power_save: bool,
    pub tts_enabled: bool,
    pub tts_model_path: String,
    pub language: String,
    pub system_prompt_template: String,
    pub selected_prompt_preset: String,
    pub custom_prompts: Vec<CustomPromptPreset>,
}

impl Default for AppSettings {
    fn default() -> Self {
        let st = SharedState::default();
        Self::from_shared(&st)
    }
}

impl AppSettings {
    pub fn from_shared(st: &SharedState) -> Self {
        Self {
            llm_model_path: st.llm_model_path.clone(),
            slm_model_path: st.slm_model_path.clone(),
            compute_backend: st.compute_backend,
            move_to_modelos: st.move_to_modelos,
            llm_temp: st.llm_temp,
            llm_ctx_size: st.llm_ctx_size,
            sampler: st.sampler,
            show_thinking: st.show_thinking,
            power_save: st.power_save,
            tts_enabled: st.tts_enabled,
            tts_model_path: st.tts_model_path.clone(),
            language: st.language.clone(),
            system_prompt_template: st.system_prompt_template.clone(),
            selected_prompt_preset: st.selected_prompt_preset.clone(),
            custom_prompts: st.custom_prompts.clone(),
        }
    }

    pub fn apply_to_shared(&self, st: &mut SharedState) {
        if !self.llm_model_path.trim().is_empty() {
            st.llm_model_path = self.llm_model_path.clone();
        }
        if !self.slm_model_path.trim().is_empty() {
            st.slm_model_path = self.slm_model_path.clone();
        }
        st.compute_backend = self.compute_backend;
        st.move_to_modelos = self.move_to_modelos;
        st.llm_temp = self.llm_temp.clamp(0.0, 2.0);
        st.llm_ctx_size = self.llm_ctx_size.clamp(512, 32768);
        st.sampler = self.sampler;
        st.show_thinking = self.show_thinking;
        st.power_save = self.power_save;
        st.tts_enabled = self.tts_enabled;
        if !self.tts_model_path.trim().is_empty() {
            st.tts_model_path = self.tts_model_path.clone();
        }
        if !self.language.trim().is_empty() {
            st.language = self.language.clone();
        }
        if !self.system_prompt_template.trim().is_empty() {
            st.system_prompt_template = self.system_prompt_template.clone();
        }
        if !self.selected_prompt_preset.trim().is_empty() {
            st.selected_prompt_preset = self.selected_prompt_preset.clone();
        }
        st.custom_prompts = self.custom_prompts.clone();
    }

    /// Caminho padrão `settings.json` ao lado do executável atual.
    pub fn default_path() -> PathBuf {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        exe_dir.join("settings.json")
    }

    pub fn load_from_path(path: &Path) -> std::io::Result<Self> {
        let raw = fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save_to_path(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, json)
    }

    pub fn load_default_or_init() -> Self {
        let path = Self::default_path();
        Self::load_from_path(&path).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_json_roundtrip() {
        let tmp = std::env::temp_dir().join("thrediars_test_settings.json");
        let mut st = SharedState::default();
        st.llm_temp = 0.42;
        st.llm_ctx_size = 2048;
        st.sampler.top_k = 25;
        st.sampler.min_p = 0.08;
        st.power_save = true;
        st.show_thinking = true;
        st.tts_enabled = true;
        st.move_to_modelos = false;
        st.compute_backend = ComputeBackend::Gpu;
        st.selected_prompt_preset = "Meu Preset".to_string();
        st.system_prompt_template = "Olá, sou {name}.".to_string();
        st.custom_prompts.push(CustomPromptPreset {
            name: "Meu Preset".to_string(),
            prompt: "Olá, sou {name}.".to_string(),
        });

        let cfg = AppSettings::from_shared(&st);
        cfg.save_to_path(&tmp).unwrap();

        let loaded = AppSettings::load_from_path(&tmp).unwrap();
        assert!((loaded.llm_temp - 0.42).abs() < 1e-6);
        assert_eq!(loaded.llm_ctx_size, 2048);
        assert_eq!(loaded.sampler.top_k, 25);
        assert!((loaded.sampler.min_p - 0.08).abs() < 1e-6);
        assert!(loaded.power_save);
        assert!(loaded.show_thinking);
        assert!(loaded.tts_enabled);
        assert!(!loaded.move_to_modelos);
        assert_eq!(loaded.compute_backend, ComputeBackend::Gpu);
        assert_eq!(loaded.selected_prompt_preset, "Meu Preset");
        assert_eq!(loaded.system_prompt_template, "Olá, sou {name}.");
        assert_eq!(loaded.custom_prompts.len(), 1);

        let _ = fs::remove_file(&tmp);
    }
}

