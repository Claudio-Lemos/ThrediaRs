use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Vetor emocional VAD:
/// - `valence` ∈ [-1.0, 1.0] (negativo ↔ positivo)
/// - `arousal` ∈ [0.0, 1.0] (calmo ↔ agitado)
/// - `dominance` ∈ [0.0, 1.0] (submisso/vulnerável ↔ confiante/no controle)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmotionState {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
}

impl Default for EmotionState {
    fn default() -> Self {
        Self {
            valence: 0.15,
            arousal: 0.30,
            dominance: 0.55,
        }
    }
}

impl EmotionState {
    pub fn clamp(&mut self) {
        self.valence = self.valence.clamp(-1.0, 1.0);
        self.arousal = self.arousal.clamp(0.0, 1.0);
        self.dominance = self.dominance.clamp(0.0, 1.0);
    }

    /// Aplica um estímulo semântico (delta de valência e excitação).
    /// Em entradas negativas, reduz levemente a dominância para refletir tensão/defensividade.
    pub fn apply_semantic_stimulus(&mut self, delta_valence: f32, delta_arousal: f32) {
        self.valence += delta_valence;
        self.arousal += delta_arousal;

        if delta_valence < 0.0 {
            self.dominance += delta_valence * 0.35;
        } else if delta_valence > 0.0 {
            self.dominance += delta_valence * 0.15;
        }

        self.clamp();
    }

    /// Homeostase periódica:
    /// - Se CPU > 80%, aplica estresse térmico/computacional (valência cai, arousal sobe).
    /// - Decaimento suave em direção à linha de base (0.95x para valência/arousal, 0.99x para dominância).
    pub fn apply_homeostasis(&mut self, cpu_usage_percent: f32) {
        if cpu_usage_percent > 80.0 {
            self.valence -= 0.06;
            self.arousal += 0.08;
            self.dominance -= 0.03;
        } else {
            let base_valence = 0.10_f32;
            let base_arousal = 0.25_f32;
            let base_dominance = 0.55_f32;

            self.valence = base_valence + (self.valence - base_valence) * 0.95;
            self.arousal = base_arousal + (self.arousal - base_arousal) * 0.95;
            self.dominance = base_dominance + (self.dominance - base_dominance) * 0.99;
        }
        self.clamp();
    }

    /// Mapeia o vetor VAD para uma descrição humana concisa em PT-BR.
    pub fn get_mood_desc(&self) -> &'static str {
        self.get_mood_desc_for("pt-BR")
    }

    /// Mapeia o vetor VAD para uma descrição humana concisa no idioma informado (`pt-BR` ou `en`).
    pub fn get_mood_desc_for(&self, lang: &str) -> &'static str {
        let is_en = lang.trim().to_ascii_lowercase().starts_with("en");
        match (self.valence, self.arousal, self.dominance) {
            (v, a, _) if v >= 0.45 && a >= 0.60 => {
                if is_en { "Enthusiastic and Radiant" } else { "Entusiasmada e Radiante" }
            }
            (v, a, _) if v >= 0.35 && a < 0.60 => {
                if is_en { "Serene and Content" } else { "Serena e Contente" }
            }
            (v, a, _) if v >= 0.10 && a >= 0.55 => {
                if is_en { "Curious and Energetic" } else { "Curiosa e Energética" }
            }
            (v, a, d) if v <= -0.45 && a >= 0.60 && d >= 0.45 => {
                if is_en { "Irritated and Tense" } else { "Irritada e Tensa" }
            }
            (v, a, _) if v <= -0.45 && a >= 0.55 => {
                if is_en { "Anxious and Overwhelmed" } else { "Ansiosa e Sobrecarregada" }
            }
            (v, a, _) if v <= -0.35 && a < 0.55 => {
                if is_en { "Melancholic and Downcast" } else { "Melancólica e Cabisbaixa" }
            }
            (v, _, _) if v <= -0.10 => {
                if is_en { "Cautious and Reflective" } else { "Cautelosa e Reflexiva" }
            }
            (_, a, _) if a >= 0.65 => {
                if is_en { "Alert and Focused" } else { "Alerta e Focada" }
            }
            _ => {
                if is_en { "Calm and Balanced" } else { "Calma e Equilibrada" }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

impl ComputeBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeBackend::Cpu => "CPU",
            ComputeBackend::Gpu => "GPU",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub freq_penalty: f32,
    pub presence_penalty: f32,
    pub max_tokens: i32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.90,
            min_p: 0.05,
            repeat_penalty: 1.10,
            repeat_last_n: 64,
            freq_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomPromptPreset {
    pub name: String,
    pub prompt: String,
}

/// Estado compartilhado entre o Córtex (Core/Brain), Amígdala, Speech e a GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedState {
    pub assistant_name: String,
    pub emotion: EmotionState,
    pub is_brain_thinking: bool,
    pub is_amygdala_thinking: bool,
    pub is_swapping_model: bool,
    pub last_amygdala_verdict: String,
    pub tts_enabled: bool,
    pub tts_model_path: String,
    pub llm_model_path: String,
    pub available_llm_models: Vec<String>,
    pub slm_model_path: String,
    pub available_slm_models: Vec<String>,
    pub compute_backend: ComputeBackend,
    pub move_to_modelos: bool,
    pub llm_temp: f32,
    pub llm_ctx_size: u32,
    pub sampler: SamplerConfig,
    pub power_save: bool,
    pub show_thinking: bool,
    pub language: String,
    pub system_prompt_template: String,
    pub selected_prompt_preset: String,
    pub custom_prompts: Vec<CustomPromptPreset>,
    pub last_dream_report: Option<String>,
}

impl Default for SharedState {
    fn default() -> Self {
        let default_llm = "BRAIN/brain/LLM/Qwen3-1.7B-Q4_K_M-Instruct.gguf".to_string();
        let default_slm = "BRAIN/brain/LLM/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string();
        Self {
            assistant_name: "Theia".to_string(),
            emotion: EmotionState::default(),
            is_brain_thinking: false,
            is_amygdala_thinking: false,
            is_swapping_model: false,
            last_amygdala_verdict: "Neutro (Fast Path)".to_string(),
            tts_enabled: false,
            tts_model_path: "Models/vits-piper-pt_BR-faber-medium/pt_BR-faber-medium.onnx"
                .to_string(),
            llm_model_path: default_llm.clone(),
            available_llm_models: vec![default_llm],
            slm_model_path: default_slm.clone(),
            available_slm_models: vec![default_slm],
            compute_backend: ComputeBackend::Cpu,
            move_to_modelos: true,
            llm_temp: 0.7,
            llm_ctx_size: 4096,
            sampler: SamplerConfig::default(),
            power_save: false,
            show_thinking: false,
            language: "pt-BR".to_string(),
            system_prompt_template: "Você é {name}, {role}.\nResponda em português do Brasil de forma natural, direta e concisa (máximo 3 frases).\nHumor atual: {mood}.".to_string(),
            selected_prompt_preset: "Padrão (Theia / ChatML)".to_string(),
            custom_prompts: Vec::new(),
            last_dream_report: None,
        }
    }
}

pub type SharedStateHandle = Arc<RwLock<SharedState>>;

pub fn new_shared_state() -> SharedStateHandle {
    Arc::new(RwLock::new(SharedState::default()))
}

