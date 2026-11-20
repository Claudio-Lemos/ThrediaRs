pub mod facts;
pub mod personality;
pub mod state;

use anyhow::Result;
use evolve::{EvolveMonitor, HealthLevel, PerfTimer};
use memchr::memchr;
use rust_stemmers::{Algorithm, Stemmer};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

pub use facts::{
    extract_declarative_facts, parse_learn_tags, parse_slm_kv_lines, ExtractedFact,
    RelevanceAssessment, RelevanceSystem, MIN_FACT_RELEVANCE,
};
pub use personality::{
    builtin_prompt_presets, find_builtin_preset_for_prompt, ModelPromptPreset, PromptComposer,
    PromptFormat,
};
pub use state::{
    new_shared_state, ComputeBackend, CustomPromptPreset, EmotionState, SamplerConfig, SharedState,
    SharedStateHandle,
};

#[derive(Debug, Clone)]
pub struct SentimentVerdict {
    pub delta_valence: f32,
    pub delta_arousal: f32,
    pub label: String,
    pub used_fast_path: bool,
}

/// Sistema 1 (Instinto): Fast Path léxico O(n) com `memchr` (`!`/`?`),
/// `rust-stemmers` (Português) e `Cow<str>`, mais extração de fatos com Sistema de Relevância
/// (Fast Path declarativo + SLM `.gguf` opcional + `[LEARN: S | A | V]`).
pub struct Amygdala {
    stemmer: Stemmer,
    model_path: Option<PathBuf>,
    #[cfg(feature = "llama")]
    runtime: Option<parking_lot::Mutex<llama_engine::LlamaRuntime>>,
}

impl Amygdala {
    pub fn new(model_path: Option<PathBuf>) -> Self {
        #[cfg(feature = "llama")]
        let runtime = {
            let is_gguf = model_path.as_ref().is_some_and(|p| {
                p.exists()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            });
            if is_gguf {
                if let Some(ref path) = model_path {
                    if let Ok(mut rt) = llama_engine::LlamaRuntime::new(1024) {
                        if rt.load_model(path, 1024, ComputeBackend::Cpu).is_ok() {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Info,
                                "brain",
                                format!("SLM da Amígdala carregada via llama-cpp-2: {:?}", path),
                            );
                            Some(parking_lot::Mutex::new(rt))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        Self {
            stemmer: Stemmer::create(Algorithm::Portuguese),
            model_path,
            #[cfg(feature = "llama")]
            runtime,
        }
    }

    pub fn has_local_slm(&self) -> bool {
        self.model_path.as_ref().is_some_and(|p| p.exists())
    }

    pub fn model_path(&self) -> Option<&PathBuf> {
        self.model_path.as_ref()
    }

    /// Realiza hot swap do modelo da Amígdala (SLM `.gguf` ou classificador `.onnx`).
    pub fn hot_swap(&mut self, new_model_path: Option<PathBuf>) -> Result<String> {
        let _timer = PerfTimer::start("amygdala_hot_swap");
        self.model_path = new_model_path;
        let path_desc = self
            .model_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "nenhum".to_string());
        let exists = self.model_path.as_ref().is_some_and(|p| p.exists());
        let is_onnx = self
            .model_path
            .as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
        let kind = if is_onnx {
            "ONNX (tract)"
        } else {
            "SLM (GGUF)"
        };

        #[cfg(feature = "llama")]
        {
            self.runtime = None;
            if exists && !is_onnx {
                if let Some(ref path) = self.model_path {
                    if let Ok(mut rt) = llama_engine::LlamaRuntime::new(1024) {
                        rt.load_model(path, 1024, ComputeBackend::Cpu)?;
                        self.runtime = Some(parking_lot::Mutex::new(rt));
                    }
                }
            }
        }

        let status = if exists {
            format!("Hot swap da Amígdala concluído [{}]: {}", kind, path_desc)
        } else {
            format!(
                "Hot swap da Amígdala aplicado em modo Fast Path [{}]: {} (arquivo ausente em disco)",
                kind, path_desc
            )
        };
        EvolveMonitor::global().record_event(HealthLevel::Info, "brain", &status);
        Ok(status)
    }

    /// Analisa sentimento e excitação em O(n) no Fast Path léxico.
    pub fn sense_sentiment(&self, input: &str) -> SentimentVerdict {
        let _timer = PerfTimer::start("amygdala_sense");
        let bytes = input.as_bytes();
        let has_exclam = memchr(b'!', bytes).is_some();
        let has_quest = memchr(b'?', bytes).is_some();

        let lower: Cow<str> = if input.bytes().all(|b| !b.is_ascii_uppercase()) {
            Cow::Borrowed(input)
        } else {
            Cow::Owned(input.to_lowercase())
        };

        let positive_stems = [
            "obrig", "feliz", "otim", "ótim", "excelent", "am", "ador", "gost", "incrivel",
            "incrí", "maravilh", "parab", "legal", "alegri", "sucess", "bom", "boa", "perfeit",
            "ajud", "valeu",
        ];
        let negative_stems = [
            "trist", "raiv", "odi", "odei", "horrivel", "horrí", "pessim", "péssim", "ruim",
            "erro", "falh", "quebr", "irrit", "chate", "cans", "estress", "medo", "preocup",
            "droga", "problem",
        ];

        let mut pos_hits = 0_i32;
        let mut neg_hits = 0_i32;

        for raw_word in lower.split(|c: char| !c.is_alphanumeric()) {
            if raw_word.len() < 2 {
                continue;
            }
            let stemmed = self.stemmer.stem(raw_word);
            if positive_stems.iter().any(|p| stemmed.starts_with(p)) {
                pos_hits += 1;
            }
            if negative_stems.iter().any(|n| stemmed.starts_with(n)) {
                neg_hits += 1;
            }
        }

        let mut dv = (pos_hits - neg_hits) as f32 * 0.22;
        let mut da = (pos_hits + neg_hits) as f32 * 0.10;
        if has_exclam {
            da += 0.18;
        }
        if has_quest {
            da += 0.08;
        }

        dv = dv.clamp(-0.65, 0.65);
        da = da.clamp(0.0, 0.50);

        let label = if dv > 0.12 {
            format!("Positivo ({:+.2}) [Fast Path]", dv)
        } else if dv < -0.12 {
            format!("Negativo ({:+.2}) [Fast Path]", dv)
        } else if has_quest {
            "Curioso / Pergunta [Fast Path]".to_string()
        } else {
            "Neutro [Fast Path]".to_string()
        };

        SentimentVerdict {
            delta_valence: dv,
            delta_arousal: da,
            label,
            used_fast_path: true,
        }
    }

    /// Avalia a relevância da entrada do usuário antes de extrair e persistir fatos.
    pub fn assess_relevance(&self, user_input: &str) -> RelevanceAssessment {
        RelevanceSystem::assess_user_input(user_input)
    }

    /// Extrai fatos usando a SLM da Amígdala (quando carregada) para decidir o que é relevante extrair,
    /// com fallback para o Fast Path declarativo quando não houver SLM em disco.
    /// Todos os candidatos passam pelo `RelevanceSystem::score_and_filter`.
    pub fn extract_facts(&self, user_input: &str, assistant_output: &str) -> Vec<ExtractedFact> {
        let _timer = PerfTimer::start("amygdala_extract_facts");
        let mut candidates = parse_learn_tags(assistant_output, 0.95);

        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            return RelevanceSystem::score_and_filter(candidates);
        }

        #[cfg(feature = "llama")]
        if let Some(ref rt_mutex) = self.runtime {
            let mut rt = rt_mutex.lock();
            if rt.is_loaded() {
                let is_tinyllama = self
                    .model_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_lowercase().contains("tinyllama"));

                let sys_inst = "Você extrai fatos pessoais do usuário. Responda apenas com linhas no formato chave: valor (por exemplo: nome, localizacao, profissao, interesse, projeto, idade) ou NONE se não houver fatos.";
                let user_msg = format!(
                    "Extraia os fatos pessoais desta frase: \"{}\"",
                    trimmed.replace('"', "'")
                );

                let slm_prompt = if is_tinyllama {
                    format!(
                        "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                        sys_inst, user_msg
                    )
                } else {
                    format!(
                        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                        sys_inst, user_msg
                    )
                };
                let sampler = SamplerConfig {
                    max_tokens: 64,
                    ..SamplerConfig::default()
                };
                match rt.generate_stream(&slm_prompt, 0.0, &sampler, |_| {}) {
                    Ok(slm_out) => {
                        let slm_clean = slm_out.trim();
                        let slm_facts = parse_slm_kv_lines(slm_clean, 0.90);
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "brain",
                            format!(
                                "[AMYGDALA-SLM] {} fato(s) extraído(s) pela SLM: {}",
                                slm_facts.len(),
                                slm_clean.replace('\n', " | ")
                            ),
                        );
                        if slm_facts.is_empty() && self.assess_relevance(trimmed).is_relevant {
                            candidates.extend(extract_declarative_facts(trimmed));
                        } else {
                            candidates.extend(slm_facts);
                        }
                    }
                    Err(e) => {
                        EvolveMonitor::global().record_event(
                            HealthLevel::Warn,
                            "brain",
                            format!("[AMYGDALA-SLM] Erro na inferência SLM: {}", e),
                        );
                    }
                }
                return RelevanceSystem::score_and_filter(candidates);
            }
        }

        // Fallback Fast Path quando nenhuma SLM `.gguf` está carregada na Amígdala
        let relevance = self.assess_relevance(trimmed);
        if relevance.is_relevant {
            candidates.extend(extract_declarative_facts(trimmed));
        }

        RelevanceSystem::score_and_filter(candidates)
    }
}

/// Descobre arquivos `.gguf` disponíveis em um diretório (ex.: `BRAIN/brain/LLM`).
pub fn discover_gguf_models(dir: &Path) -> Vec<String> {
    discover_models_with_exts(dir, &["gguf"])
}

/// Descobre modelos para a Amígdala (`.gguf` ou `.onnx`) disponíveis em um diretório.
pub fn discover_amygdala_models(dir: &Path) -> Vec<String> {
    discover_models_with_exts(dir, &["gguf", "onnx"])
}

fn discover_models_with_exts(dir: &Path, exts: &[&str]) -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.iter().any(|allowed| ext.eq_ignore_ascii_case(allowed)) {
                        models.push(path.display().to_string());
                    }
                }
            }
        }
    }
    models.sort();
    models
}

#[cfg(feature = "llama")]
mod llama_engine {
    use anyhow::{anyhow, Result};
    use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::sampling::LlamaSampler;
    use self_cell::self_cell;
    use std::num::NonZeroU32;
    use std::path::Path;

    use crate::state::{ComputeBackend, SamplerConfig};

    pub struct LlamaOwner {
        pub model: LlamaModel,
    }

    self_cell! {
        pub struct LlamaCell {
            owner: LlamaOwner,
            #[covariant]
            dependent: LlamaContext,
        }
    }

    static SHARED_LLAMA_BACKEND: std::sync::OnceLock<std::result::Result<LlamaBackend, String>> =
        std::sync::OnceLock::new();

    fn shared_llama_backend() -> Result<&'static LlamaBackend> {
        let res = SHARED_LLAMA_BACKEND.get_or_init(|| {
            llama_cpp_2::send_logs_to_tracing(
                llama_cpp_2::LogOptions::default().with_logs_enabled(false),
            );
            LlamaBackend::init().map_err(|e| format!("Falha ao inicializar LlamaBackend: {:?}", e))
        });
        res.as_ref().map_err(|e| anyhow!("{}", e))
    }

    pub struct LlamaRuntime {
        backend: &'static LlamaBackend,
        cell: Option<LlamaCell>,
        n_ctx: u32,
        n_past: i32,
    }

    impl LlamaRuntime {
        pub fn new(n_ctx: u32) -> Result<Self> {
            let backend = shared_llama_backend()?;
            Ok(Self {
                backend,
                cell: None,
                n_ctx: n_ctx.max(512),
                n_past: 0,
            })
        }

        pub fn load_model(
            &mut self,
            path: &Path,
            n_ctx: u32,
            compute_backend: ComputeBackend,
        ) -> Result<()> {
            self.cell = None;
            self.n_past = 0;
            self.n_ctx = n_ctx.max(512);

            let mut model_params = LlamaModelParams::default();
            if compute_backend == ComputeBackend::Gpu {
                model_params = model_params.with_n_gpu_layers(99);
            } else {
                model_params = model_params.with_n_gpu_layers(0);
            }

            let model = LlamaModel::load_from_file(self.backend, path, &model_params)
                .map_err(|e| anyhow!("Erro ao carregar GGUF {:?}: {:?}", path, e))?;

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(self.n_ctx))
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);

            let backend_ref = self.backend;
            let cell = LlamaCell::try_new(LlamaOwner { model }, |owner| {
                owner
                    .model
                    .new_context(backend_ref, ctx_params)
                    .map_err(|e| anyhow!("Erro ao criar LlamaContext: {:?}", e))
            })?;

            self.cell = Some(cell);
            Ok(())
        }

        pub fn is_loaded(&self) -> bool {
            self.cell.is_some()
        }

        pub fn clear_kv(&mut self) {
            if let Some(cell) = self.cell.as_mut() {
                cell.with_dependent_mut(|_, ctx| {
                    let _ = ctx.clear_kv_cache_seq(Some(0), None, None);
                });
            }
            self.n_past = 0;
        }

        pub fn generate_stream<F>(
            &mut self,
            prompt: &str,
            temperature: f32,
            sampler_cfg: &SamplerConfig,
            mut on_token: F,
        ) -> Result<String>
        where
            F: FnMut(&str),
        {
            let cell = self
                .cell
                .as_mut()
                .ok_or_else(|| anyhow!("Nenhum modelo GGUF carregado no LlamaRuntime"))?;

            let mut output = String::new();
            let n_ctx_i32 = self.n_ctx as i32;

            cell.with_dependent_mut(|owner, ctx| -> Result<()> {
                // Limpa KV Cache para evitar estouro de n_ctx e garantir prompt ChatML limpo
                let _ = ctx.clear_kv_cache_seq(Some(0), None, None);

                let vocab = owner.model.vocab();
                // ChatML já começa com <|im_start|>, então add_special=false evita BOS duplicado;
                // modelos como TinyLlama (<|system|>) precisam de BOS (add_special=true).
                let add_special = !prompt.starts_with("<|im_start|>");
                let tokens = vocab.tokenize(prompt.as_bytes(), add_special, true);

                if tokens.is_empty() {
                    return Ok(());
                }

                let max_prompt = (n_ctx_i32 - 64).max(128) as usize;
                let prompt_slice = if tokens.len() > max_prompt {
                    &tokens[tokens.len() - max_prompt..]
                } else {
                    &tokens[..]
                };

                let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
                let last_idx = (prompt_slice.len() - 1) as i32;
                for (i, tok) in prompt_slice.iter().enumerate() {
                    batch
                        .add(*tok, i as i32, &[0], i as i32 == last_idx)
                        .map_err(|e| anyhow!("Batch add error: {:?}", e))?;
                }

                ctx.decode(&mut batch)
                    .map_err(|e| anyhow!("Decode prompt error: {:?}", e))?;

                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(42);
                let mut sampler = LlamaSampler::chain_simple([
                    LlamaSampler::penalties(
                        owner.model.n_vocab(),
                        sampler_cfg.repeat_last_n.max(1),
                        sampler_cfg.repeat_penalty.max(1.0),
                        sampler_cfg.freq_penalty.max(0.0),
                        sampler_cfg.presence_penalty.max(0.0),
                    ),
                    LlamaSampler::top_k(sampler_cfg.top_k.max(1)),
                    LlamaSampler::top_p(sampler_cfg.top_p.clamp(0.05, 1.0), 1),
                    LlamaSampler::min_p(sampler_cfg.min_p.clamp(0.0, 1.0), 1),
                    LlamaSampler::temp(temperature.max(0.01)),
                    LlamaSampler::dist(seed),
                ]);

                let mut n_cur = prompt_slice.len() as i32;
                let max_gen = (n_ctx_i32 - n_cur)
                    .min(sampler_cfg.max_tokens.max(16))
                    .max(16);
                let mut utf8_buf: Vec<u8> = Vec::new();
                let stop_markers = [
                    "<|im_end|>",
                    "<|im_start|>",
                    "< |im_end",
                    "< |im_start",
                    "<|endoftext|>",
                    "<|eot_id|>",
                    "<|end_of_text|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<end_of_turn>",
                    "</s>",
                ];

                for _ in 0..max_gen {
                    let token = sampler.sample(ctx, batch.n_tokens() - 1);
                    sampler.accept(token);

                    if vocab.is_eog(token) {
                        break;
                    }

                    let bytes = vocab.token_to_piece(token, false, None);
                    if !bytes.is_empty() {
                        utf8_buf.extend_from_slice(&bytes);
                        if let Ok(valid_str) = std::str::from_utf8(&utf8_buf) {
                            if stop_markers.iter().any(|m| valid_str.contains(m)) {
                                break;
                            }
                            output.push_str(valid_str);
                            if let Some(pos) =
                                stop_markers.iter().filter_map(|m| output.find(m)).min()
                            {
                                output.truncate(pos);
                                break;
                            }
                            on_token(valid_str);
                            utf8_buf.clear();
                        }
                    }

                    batch.clear();
                    batch
                        .add(token, n_cur, &[0], true)
                        .map_err(|e| anyhow!("Batch add gen error: {:?}", e))?;
                    n_cur += 1;

                    ctx.decode(&mut batch)
                        .map_err(|e| anyhow!("Decode gen error: {:?}", e))?;
                }

                // Flush UTF-8 residual
                if !utf8_buf.is_empty() {
                    let lossy = String::from_utf8_lossy(&utf8_buf).into_owned();
                    output.push_str(&lossy);
                    on_token(&lossy);
                }

                Ok(())
            })?;

            self.n_past = 0;
            Ok(output)
        }
    }
}

/// Sistema 2 (LLM): motor de inferência com streaming token-a-token e gerenciamento de contexto.
/// Quando `feature = "llama"` e o arquivo `.gguf` estiverem presentes, integra `llama-cpp-2`.
/// Caso contrário, opera em modo Protótipo Cognitivo Local totalmente funcional.
pub struct Brain {
    model_path: Option<PathBuf>,
    n_ctx: u32,
    backend: ComputeBackend,
    kv_prefix_len: usize,
    #[cfg(feature = "llama")]
    runtime: Option<llama_engine::LlamaRuntime>,
}

impl Brain {
    pub fn new(model_path: Option<PathBuf>, n_ctx: u32) -> Self {
        #[allow(unused_mut)]
        let mut brain = Self {
            model_path: model_path.clone(),
            n_ctx: n_ctx.max(512),
            backend: ComputeBackend::Cpu,
            kv_prefix_len: 0,
            #[cfg(feature = "llama")]
            runtime: llama_engine::LlamaRuntime::new(n_ctx).ok(),
        };

        let loaded_gguf = model_path.as_ref().is_some_and(|p| p.exists());
        #[cfg(feature = "llama")]
        if loaded_gguf {
            if let (Some(ref path), Some(ref mut rt)) = (&model_path, &mut brain.runtime) {
                match rt.load_model(path, brain.n_ctx, brain.backend) {
                    Ok(()) => {
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "brain",
                            format!("Modelo GGUF carregado via llama-cpp-2: {:?}", path),
                        );
                    }
                    Err(e) => {
                        EvolveMonitor::global().record_event(
                            HealthLevel::Error,
                            "brain",
                            format!("Falha ao carregar GGUF {:?}: {}", path, e),
                        );
                    }
                }
            }
        }

        let mode = if cfg!(feature = "llama") && loaded_gguf {
            "llama.cpp ativo (GGUF carregado)"
        } else if loaded_gguf {
            "GGUF detectado (compile com `--features llama` + `cmake` para inferência nativa)"
        } else {
            "Protótipo Cognitivo Local (sem modelo .gguf em disco)"
        };
        EvolveMonitor::global().record_event(
            HealthLevel::Info,
            "brain",
            format!("Brain inicializado | n_ctx={} | modo={}", brain.n_ctx, mode),
        );
        brain
    }

    pub fn model_path(&self) -> Option<&PathBuf> {
        self.model_path.as_ref()
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// Realiza hot swap do modelo LLM em tempo de execução sem reiniciar a aplicação.
    /// Limpa o KV Cache, atualiza `n_ctx` e backend (`CPU`/`GPU`), e recarrega o modelo.
    pub fn hot_swap(
        &mut self,
        new_model_path: Option<PathBuf>,
        n_ctx: u32,
        backend: ComputeBackend,
    ) -> Result<String> {
        let _timer = PerfTimer::start("brain_hot_swap");
        self.kv_prefix_len = 0;
        self.n_ctx = n_ctx.max(512);
        self.backend = backend;
        self.model_path = new_model_path;

        let path_desc = self
            .model_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "nenhum".to_string());
        let exists = self.model_path.as_ref().is_some_and(|p| p.exists());

        #[cfg(feature = "llama")]
        if exists {
            if let (Some(ref path), Some(ref mut rt)) = (&self.model_path, &mut self.runtime) {
                rt.load_model(path, self.n_ctx, self.backend)?;
            }
        }

        let status = if exists && cfg!(feature = "llama") {
            format!(
                "Hot swap concluído (llama.cpp ativo): {} (n_ctx={}, backend={})",
                path_desc,
                self.n_ctx,
                self.backend.as_str()
            )
        } else if exists {
            format!(
                "Modelo selecionado: {} — atenção: binário compilado sem feature `llama` (instale `cmake` e rode `cargo run -p core --features llama`).",
                path_desc
            )
        } else {
            format!(
                "Hot swap aplicado em modo protótipo local: {} (arquivo .gguf ausente, n_ctx={}, backend={})",
                path_desc,
                self.n_ctx,
                self.backend.as_str()
            )
        };

        EvolveMonitor::global().record_event(HealthLevel::Info, "brain", &status);
        Ok(status)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_prefix_len = 0;
        #[cfg(feature = "llama")]
        if let Some(ref mut rt) = self.runtime {
            rt.clear_kv();
        }
        EvolveMonitor::global().record_event(
            HealthLevel::Info,
            "brain",
            "KV Cache limpo com sucesso.",
        );
    }

    /// Gera resposta com streaming token-a-token chamando `on_token` para cada chunk UTF-8.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream<F>(
        &mut self,
        prompt_chatml: &str,
        user_input: &str,
        emotion: &EmotionState,
        rag_snippets: &[String],
        temperature: f32,
        sampler_cfg: &SamplerConfig,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let _timer = PerfTimer::start("brain_generate");

        #[cfg(feature = "llama")]
        if let Some(ref mut rt) = self.runtime {
            if rt.is_loaded() {
                return rt.generate_stream(prompt_chatml, temperature, sampler_cfg, on_token);
            }
        }

        let _ = (prompt_chatml, sampler_cfg);
        self.kv_prefix_len = self
            .kv_prefix_len
            .saturating_add(user_input.len())
            .min(self.n_ctx as usize);

        let mood = emotion.get_mood_desc();
        let lower = user_input.to_lowercase();

        let think_block = format!(
            "<think>Analisando entrada (temp={:.2}, humor={}, rag={} itens). Formulando resposta concisa em PT-BR.</think>",
            temperature,
            mood,
            rag_snippets.len()
        );

        let has_gguf_on_disk = self.model_path.as_ref().is_some_and(|p| p.exists());

        let reply_body = if has_gguf_on_disk && !cfg!(feature = "llama") {
            format!(
                "Encontrei o modelo `{}` no disco, mas este executável foi compilado sem o motor `llama-cpp-2` (requer `cmake`). Instale o `cmake` (`sudo pacman -S cmake`) e execute `cargo run -p core --features llama` para gerar respostas reais com o modelo!",
                self.model_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("modelo.gguf")
            )
        } else if lower.contains("oi")
            || lower.contains("olá")
            || lower.contains("ola")
            || lower.contains("bom dia")
            || lower.contains("boa tarde")
            || lower.contains("boa noite")
        {
            format!(
                "Olá! Estou online e me sentindo {}. Como posso ajudar você hoje?",
                mood.to_lowercase()
            )
        } else if lower.contains("como você está")
            || lower.contains("como voce esta")
            || lower.contains("tudo bem")
        {
            format!(
                "Estou {} (Valência {:+.2}, Excitação {:.2}). Todos os módulos cognitivos estão operando normalmente!",
                mood.to_lowercase(),
                emotion.valence,
                emotion.arousal
            )
        } else if lower.contains("quem é você")
            || lower.contains("quem e voce")
            || lower.contains("seu nome")
        {
            "Eu sou a Theia, sua assistente cognitiva local construída em Rust. Combino memória SQLite com FTS5, estado emocional VAD e telemetria em tempo real.".to_string()
        } else if !rag_snippets.is_empty() {
            format!(
                "Lembro disso pelas minhas memórias: {}. Considerando isso e meu estado {}, posso te ajudar com os próximos passos!",
                rag_snippets[0],
                mood.to_lowercase()
            )
        } else if lower.contains("meu nome é") || lower.contains("me chamo") {
            "Prazer em conhecer você! Vou guardar seu nome na minha memória persistente para nossas próximas conversas.".to_string()
        } else {
            format!(
                "Entendi o que você disse sobre \"{}\". Estou {} e operando no motor local — quando você adicionar um modelo .gguf em BRAIN/brain/LLM/, meu raciocínio completo ficará ainda mais rico!",
                user_input.trim(),
                mood.to_lowercase()
            )
        };

        let full_output = format!("{}{}", think_block, reply_body);

        // Streaming token-a-token preservando UTF-8
        for piece in split_stream_chunks(&full_output) {
            on_token(&piece);
            thread::sleep(Duration::from_millis(12));
        }

        Ok(full_output)
    }
}

fn split_stream_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch.is_whitespace() || ch == '>' || current.chars().count() >= 6 {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_emotion_vad_and_homeostasis() {
        let mut e = EmotionState::default();
        let initial_dom = e.dominance;
        e.apply_semantic_stimulus(-0.5, 0.3);
        assert!(e.valence < 0.0);
        assert!(e.dominance < initial_dom);

        let before_stress_arousal = e.arousal;
        e.apply_homeostasis(92.0);
        assert!(e.arousal >= before_stress_arousal);
        assert!(!e.get_mood_desc().is_empty());
    }

    #[test]
    fn test_amygdala_fast_path_and_extract_facts() {
        let am = Amygdala::new(None);
        let pos = am.sense_sentiment("Muito obrigado! Isso ficou incrível e ótimo!");
        assert!(pos.delta_valence > 0.0);
        assert!(pos.used_fast_path);

        let neg = am.sense_sentiment("Que erro horrível, estou irritado e triste!");
        assert!(neg.delta_valence < 0.0);

        let facts = am.extract_facts(
            "Oi, meu nome é Axel e eu gosto de Rust.",
            "Prazer! [LEARN: Projeto | linguagem | Rust]",
        );
        assert!(facts
            .iter()
            .any(|f| f.subject == "Projeto" && f.value == "Rust"));
        assert!(facts
            .iter()
            .any(|f| f.subject == "Usuario" && f.attribute == "nome" && f.value == "Axel"));
    }

    #[test]
    fn test_prompt_composer_chatml() {
        let mut id = HashMap::new();
        id.insert("nome".to_string(), "Theia".to_string());
        let emo = EmotionState::default();
        let prompt = PromptComposer::compose_chatml(
            &id,
            &emo,
            &["Usuario gosta de Rust".to_string()],
            "<|im_start|>user\nOi<|im_end|>\n",
            "Tudo bem?",
        );
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("Theia"));
        assert!(prompt.contains("Usuario gosta de Rust"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_brain_hot_swap_and_discover() {
        let mut brain = Brain::new(None, 4096);
        assert_eq!(brain.n_ctx(), 4096);
        assert_eq!(brain.backend(), ComputeBackend::Cpu);

        let msg = brain
            .hot_swap(
                Some(PathBuf::from("BRAIN/brain/LLM/novo-modelo.gguf")),
                2048,
                ComputeBackend::Gpu,
            )
            .unwrap();
        assert!(msg.contains("novo-modelo.gguf"));
        assert_eq!(brain.n_ctx(), 2048);
        assert_eq!(brain.backend(), ComputeBackend::Gpu);
        assert_eq!(
            brain.model_path().unwrap().to_str().unwrap(),
            "BRAIN/brain/LLM/novo-modelo.gguf"
        );

        let empty = discover_gguf_models(Path::new("/tmp/non_existent_llm_dir_12345"));
        assert!(empty.is_empty());
    }

    #[test]
    fn test_amygdala_hot_swap() {
        let mut am = Amygdala::new(None);
        let msg = am
            .hot_swap(Some(PathBuf::from("Models/amygdala_sentiment.onnx")))
            .unwrap();
        assert!(msg.contains("ONNX (tract)"));
        assert!(msg.contains("amygdala_sentiment.onnx"));
        assert_eq!(
            am.model_path().unwrap().to_str().unwrap(),
            "Models/amygdala_sentiment.onnx"
        );
    }
}
