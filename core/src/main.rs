#[macro_use]
extern crate rust_i18n;

i18n!("locales", fallback = "pt-BR");

mod commands;
mod gui;
mod settings;

use brain::{
    discover_amygdala_models, discover_gguf_models, new_shared_state, Amygdala, Brain,
    PromptComposer, SamplerConfig, SharedStateHandle,
};
use commands::{
    sanitize_assistant_text, CoreCommand, IntentDispatcher, MemorySnapshot, MemorySnapshotHandle,
    SystemIntent,
};
use eframe::egui;
use evolve::{EvolveMonitor, HealthLevel, PerfTimer};
use gui::ThrediaApp;
use memory::{is_assistant_name_fact, sync_assistant_name_if_changed, Memory};
use mimalloc::MiMalloc;
use parking_lot::{Mutex, RwLock};
use settings::AppSettings;
use speech::{SpeechEngine, SpeechHandle};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::time::{sleep, Duration};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn executable_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn resolve_exe_or_cwd_file(file_name: &str) -> PathBuf {
    let exe_candidate = executable_dir().join(file_name);
    if exe_candidate.is_file() {
        return exe_candidate;
    }
    let cwd_candidate = PathBuf::from(file_name);
    if cwd_candidate.is_file() {
        return cwd_candidate;
    }
    exe_candidate
}

fn executable_modelos_dir() -> PathBuf {
    let dir = executable_dir().join("modelos");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn move_model_to_modelos_dir(src_path: &str) -> Result<String, String> {
    let src = PathBuf::from(src_path);
    if !src.is_file() {
        return Err(format!("Arquivo não encontrado: {}", src_path));
    }
    let dir = executable_modelos_dir();
    let file_name = src
        .file_name()
        .ok_or_else(|| "Nome de arquivo inválido".to_string())?;
    let dst = dir.join(file_name);

    if src == dst {
        return Ok(dst.display().to_string());
    }

    if let Err(_rename_err) = std::fs::rename(&src, &dst) {
        std::fs::copy(&src, &dst).map_err(|e| format!("Falha ao copiar para {:?}: {}", dst, e))?;
        let _ = std::fs::remove_file(&src);
    }

    EvolveMonitor::global().record_event(
        HealthLevel::Info,
        "brain",
        format!("[MODEL_MOVE] Modelo movido para {}", dst.display()),
    );
    Ok(dst.display().to_string())
}

fn refresh_available_models(shared: &SharedStateHandle) {
    let modelos_dir = executable_modelos_dir();
    let legacy_dir = std::path::Path::new("BRAIN/brain/LLM");

    let mut found_llm = discover_gguf_models(&modelos_dir);
    for m in discover_gguf_models(legacy_dir) {
        if !found_llm.contains(&m) {
            found_llm.push(m);
        }
    }

    let mut found_slm = discover_amygdala_models(&modelos_dir);
    for m in discover_amygdala_models(legacy_dir) {
        if !found_slm.contains(&m) {
            found_slm.push(m);
        }
    }

    let (cur_llm, cur_slm) = {
        let st = shared.read();
        (st.llm_model_path.clone(), st.slm_model_path.clone())
    };

    if !cur_llm.is_empty()
        && !found_llm.contains(&cur_llm)
        && std::path::Path::new(&cur_llm).exists()
    {
        found_llm.push(cur_llm);
    }
    if found_llm.is_empty() {
        found_llm.push("BRAIN/brain/LLM/Qwen3-1.7B-Q4_K_M-Instruct.gguf".to_string());
    }

    if !cur_slm.is_empty()
        && !found_slm.contains(&cur_slm)
        && std::path::Path::new(&cur_slm).exists()
    {
        found_slm.push(cur_slm);
    }
    if found_slm.is_empty() {
        found_slm.push("BRAIN/brain/LLM/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string());
    }

    let mut st = shared.write();
    st.available_llm_models = found_llm;
    st.available_slm_models = found_slm;
}

async fn sync_memory_snapshot(memory: &Option<Memory>, mem_snap: &MemorySnapshotHandle) {
    if let Some(ref mem) = memory {
        let facts = mem.get_all_facts().await.unwrap_or_default();
        let messages = mem.get_recent_messages(100).await.unwrap_or_default();
        let mut guard = mem_snap.write();
        guard.facts = facts;
        guard.messages = messages;
    }
}

/// Pipeline assíncrono do Córtex:
/// Dispatcher -> Amígdala -> SQLite + RAG -> PromptComposer -> Brain streaming (`_TKN_:`, `_STREAM_END_`)
/// -> bufferização de sentenças para TTS (ignorando `<think>`) -> pós-processamento (salvar resposta + `extract_facts`).
#[allow(clippy::too_many_arguments)]
fn spawn_core_processor(
    rt: &tokio::runtime::Handle,
    mut cmd_rx: UnboundedReceiver<CoreCommand>,
    ui_tx: UnboundedSender<String>,
    shared: SharedStateHandle,
    memory: Option<Memory>,
    mem_snap: MemorySnapshotHandle,
    brain: Arc<Mutex<Brain>>,
    amygdala: Arc<Mutex<Amygdala>>,
    speech: SpeechHandle,
) {
    rt.spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            let raw_input = match cmd {
                CoreCommand::RefreshModels => {
                    refresh_available_models(&shared);
                    EvolveMonitor::global().record_event(
                        HealthLevel::Info,
                        "brain",
                        "Lista de modelos .gguf atualizada.",
                    );
                    continue;
                }
                CoreCommand::HotSwapLlm {
                    model_path,
                    n_ctx,
                    backend,
                    move_to_modelos,
                } => {
                    {
                        shared.write().is_swapping_model = true;
                    }
                    let brain_clone = Arc::clone(&brain);
                    let target_path = if move_to_modelos {
                        match tokio::task::spawn_blocking({
                            let src = model_path.clone();
                            move || move_model_to_modelos_dir(&src)
                        })
                        .await
                        {
                            Ok(Ok(moved)) => moved,
                            Ok(Err(e)) => {
                                EvolveMonitor::global().record_event(
                                    HealthLevel::Warn,
                                    "brain",
                                    format!("Não foi possível mover modelo: {}", e),
                                );
                                model_path.clone()
                            }
                            Err(_) => model_path.clone(),
                        }
                    } else {
                        model_path.clone()
                    };

                    let model_for_swap = target_path.clone();
                    let swap_res = tokio::task::spawn_blocking(move || {
                        brain_clone.lock().hot_swap(
                            Some(PathBuf::from(model_for_swap)),
                            n_ctx,
                            backend,
                        )
                    })
                    .await;

                    {
                        let mut st = shared.write();
                        st.is_swapping_model = false;
                        if let Ok(Ok(_)) = swap_res {
                            st.llm_model_path = target_path.clone();
                            st.llm_ctx_size = n_ctx;
                            st.compute_backend = backend;
                        }
                    }
                    refresh_available_models(&shared);

                    match swap_res {
                        Ok(Ok(msg)) => {
                            let _ = ui_tx.send(format!("_SYS_:{}", msg));
                        }
                        Ok(Err(e)) => {
                            let err = format!("Falha no hot swap do modelo: {}", e);
                            EvolveMonitor::global().record_event(HealthLevel::Error, "brain", &err);
                            let _ = ui_tx.send(format!("_SYS_:{}", err));
                        }
                        Err(e) => {
                            let err = format!("Task de hot swap falhou: {}", e);
                            EvolveMonitor::global().record_event(HealthLevel::Error, "core", &err);
                            let _ = ui_tx.send(format!("_SYS_:{}", err));
                        }
                    }
                    continue;
                }
                CoreCommand::HotSwapSlm {
                    model_path,
                    move_to_modelos,
                } => {
                    {
                        shared.write().is_swapping_model = true;
                    }
                    let am_clone = Arc::clone(&amygdala);
                    let target_path = if move_to_modelos {
                        match tokio::task::spawn_blocking({
                            let src = model_path.clone();
                            move || move_model_to_modelos_dir(&src)
                        })
                        .await
                        {
                            Ok(Ok(moved)) => moved,
                            Ok(Err(e)) => {
                                EvolveMonitor::global().record_event(
                                    HealthLevel::Warn,
                                    "brain",
                                    format!("Não foi possível mover SLM: {}", e),
                                );
                                model_path.clone()
                            }
                            Err(_) => model_path.clone(),
                        }
                    } else {
                        model_path.clone()
                    };

                    let model_for_swap = target_path.clone();
                    let swap_res = tokio::task::spawn_blocking(move || {
                        am_clone.lock().hot_swap(Some(PathBuf::from(model_for_swap)))
                    })
                    .await;

                    {
                        let mut st = shared.write();
                        st.is_swapping_model = false;
                        if let Ok(Ok(_)) = swap_res {
                            st.slm_model_path = target_path.clone();
                        }
                    }
                    refresh_available_models(&shared);

                    match swap_res {
                        Ok(Ok(msg)) => {
                            let _ = ui_tx.send(format!("_SYS_:{}", msg));
                        }
                        Ok(Err(e)) => {
                            let err = format!("Falha no hot swap da SLM: {}", e);
                            EvolveMonitor::global().record_event(HealthLevel::Error, "brain", &err);
                            let _ = ui_tx.send(format!("_SYS_:{}", err));
                        }
                        Err(e) => {
                            let err = format!("Task de hot swap da SLM falhou: {}", e);
                            EvolveMonitor::global().record_event(HealthLevel::Error, "core", &err);
                            let _ = ui_tx.send(format!("_SYS_:{}", err));
                        }
                    }
                    continue;
                }
                CoreCommand::RefreshMemoryView => {
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::ExportDiag => {
                    match EvolveMonitor::global().export_diag_json() {
                        Ok(()) => {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Info,
                                "evolve",
                                "Diag.json exportado manualmente.",
                            );
                        }
                        Err(e) => {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Error,
                                "evolve",
                                format!("Falha ao exportar Diag.json: {}", e),
                            );
                        }
                    }
                    continue;
                }
                CoreCommand::RunDreamReport => {
                    let brain_clone = Arc::clone(&brain);
                    let shared_clone = Arc::clone(&shared);
                    tokio::spawn(async move {
                        run_blind_vault_dream(brain_clone, shared_clone, true).await;
                    });
                    continue;
                }
                CoreCommand::SaveSettings => {
                    let cfg = AppSettings::from_shared(&shared.read());
                    let path = AppSettings::default_path();
                    if let Err(e) = cfg.save_to_path(&path) {
                        EvolveMonitor::global().record_event(
                            HealthLevel::Error,
                            "core",
                            format!("Falha ao salvar configurações: {}", e),
                        );
                    }
                    continue;
                }
                CoreCommand::AddFact {
                    subject,
                    attribute,
                    value,
                    confidence,
                } => {
                    if let Some(ref mem) = memory {
                        let old_name = shared.read().assistant_name.clone();
                        if let Err(e) = mem.learn_fact(&subject, &attribute, &value, confidence).await {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Error,
                                "memory",
                                format!("Erro ao adicionar fato: {}", e),
                            );
                        } else {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Info,
                                "memory",
                                format!("[MEM_ADD] {} | {} = {}", subject, attribute, value),
                            );
                            if let Ok(n) = sync_assistant_name_if_changed!(
                                mem,
                                &subject,
                                &attribute,
                                &old_name,
                                &value,
                                100
                            ) {
                                if is_assistant_name_fact(&subject, &attribute) {
                                    shared.write().assistant_name = value.trim().to_string();
                                    let _ = ui_tx.send(format!("_RENAME_:{}|{}", old_name, value.trim()));
                                    if n > 0 {
                                        EvolveMonitor::global().record_event(
                                            HealthLevel::Info,
                                            "memory",
                                            format!("[NAME_SYNC] {} referências recentes atualizadas ({} -> {})", n, old_name, value.trim()),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::UpdateFact {
                    id,
                    subject,
                    attribute,
                    value,
                    confidence,
                } => {
                    if let Some(ref mem) = memory {
                        let old_name = shared.read().assistant_name.clone();
                        if let Err(e) = mem
                            .update_fact(id, &subject, &attribute, &value, confidence)
                            .await
                        {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Error,
                                "memory",
                                format!("Erro ao atualizar fato #{}: {}", id, e),
                            );
                        } else {
                            EvolveMonitor::global().record_event(
                                HealthLevel::Info,
                                "memory",
                                format!("[MEM_EDIT] #{} -> {} | {} = {}", id, subject, attribute, value),
                            );
                            if let Ok(n) = sync_assistant_name_if_changed!(
                                mem,
                                &subject,
                                &attribute,
                                &old_name,
                                &value,
                                100
                            ) {
                                if is_assistant_name_fact(&subject, &attribute) {
                                    shared.write().assistant_name = value.trim().to_string();
                                    let _ = ui_tx.send(format!("_RENAME_:{}|{}", old_name, value.trim()));
                                    if n > 0 {
                                        EvolveMonitor::global().record_event(
                                            HealthLevel::Info,
                                            "memory",
                                            format!("[NAME_SYNC] {} referências recentes atualizadas ({} -> {})", n, old_name, value.trim()),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::DeleteFact { id } => {
                    if let Some(ref mem) = memory {
                        let _ = mem.delete_fact(id).await;
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "memory",
                            format!("[MEM_DEL] Fato #{} removido", id),
                        );
                    }
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::DeleteChatMessage { id } => {
                    if let Some(ref mem) = memory {
                        let _ = mem.delete_chat_message(id).await;
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "memory",
                            format!("[MEM_DEL] Mensagem #{} removida", id),
                        );
                    }
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::ClearChatHistory => {
                    if let Some(ref mem) = memory {
                        let _ = mem.clear_chat_history().await;
                    }
                    brain.lock().clear_kv_cache();
                    sync_memory_snapshot(&memory, &mem_snap).await;
                    continue;
                }
                CoreCommand::UserMessage(text) => text,
            };

            let input = raw_input.trim().to_string();
            if input.is_empty() {
                continue;
            }

            // 1. IntentDispatcher (Zero-Latency Regex)
            if let Some(intent) = IntentDispatcher::dispatch(&input) {
                let reply = match intent {
                    SystemIntent::Status => {
                        let snap = EvolveMonitor::global().snapshot();
                        let st = shared.read().clone();
                        let reflex = body::evaluate_emotional_reflex(&st.emotion);
                        format!(
                            "{}\n• CPU: {:.1}% | RAM: {}/{} MB (Proc: {} MB) | Temp: {:.1}°C\n• {}: {} (V={:+.2}, A={:.2}, D={:.2}) | {}: {:?}\n• {}: {}",
                            t!("cmd.status_title"),
                            snap.cpu_usage_percent,
                            snap.used_memory_mb,
                            snap.total_memory_mb,
                            snap.process_memory_mb,
                            snap.max_temp_celsius,
                            t!("cmd.status_mood"),
                            st.emotion.get_mood_desc_for(&st.language),
                            st.emotion.valence,
                            st.emotion.arousal,
                            st.emotion.dominance,
                            t!("cmd.status_reflex"),
                            reflex,
                            t!("cmd.status_amygdala"),
                            st.last_amygdala_verdict
                        )
                    }
                    SystemIntent::DebugMemory => {
                        if let Some(ref mem) = memory {
                            match mem.get_all_facts().await {
                                Ok(facts) if !facts.is_empty() => {
                                    let mut out = format!("{}\n", t!("cmd.memories_header", count = facts.len()));
                                    for f in facts.iter().take(25) {
                                        out.push_str(&format!(
                                            "• #{} [{}] {} = {} (conf={:.2})\n",
                                            f.id, f.subject, f.attribute, f.value, f.confidence
                                        ));
                                    }
                                    out
                                }
                                Ok(_) => t!("cmd.no_facts").to_string(),
                                Err(e) => t!("cmd.memory_query_error", err = e).to_string(),
                            }
                        } else {
                            t!("cmd.memory_unavailable").to_string()
                        }
                    }
                    SystemIntent::ClearHistory => {
                        if let Some(ref mem) = memory {
                            let _ = mem.clear_chat_history().await;
                        }
                        brain.lock().clear_kv_cache();
                        sync_memory_snapshot(&memory, &mem_snap).await;
                        t!("cmd.history_cleared").to_string()
                    }
                    SystemIntent::Help => {
                        let lang = shared.read().language.clone();
                        IntentDispatcher::help_text_for(&lang).to_string()
                    }
                };

                let _ = ui_tx.send(format!("_TKN_:{}", reply));
                let _ = ui_tx.send("_STREAM_END_".to_string());
                continue;
            }

            // 2. Amígdala (Sistema 1 - sense_sentiment)
            {
                let mut st = shared.write();
                st.is_amygdala_thinking = true;
                st.is_brain_thinking = true;
            }

            let verdict = {
                let am = amygdala.lock();
                am.sense_sentiment(&input)
            };

            let (emotion_now, llm_temp, sampler_cfg, lang_now, prompt_tpl, prompt_preset) = {
                let mut st = shared.write();
                st.emotion
                    .apply_semantic_stimulus(verdict.delta_valence, verdict.delta_arousal);
                st.last_amygdala_verdict = verdict.label.clone();
                st.is_amygdala_thinking = false;
                (
                    st.emotion,
                    st.llm_temp,
                    st.sampler,
                    st.language.clone(),
                    st.system_prompt_template.clone(),
                    st.selected_prompt_preset.clone(),
                )
            };

            // 3. Buscar histórico ChatML & contexto RAG (FTS5) antes de persistir o input atual (evita duplicação no prompt)
            let (identity, history_chatml, rag_snippets) = if let Some(ref mem) = memory {
                let id = mem.get_identity("Assistente").await.unwrap_or_default();
                let hist = mem.get_recent_chat(10).await.unwrap_or_default();
                let hits = mem.search_memories(&input, 5).await.unwrap_or_default();
                let rag: Vec<String> = hits
                    .into_iter()
                    .map(|f| format!("{} -> {}: {}", f.subject, f.attribute, f.value))
                    .collect();
                let _ = mem.add_message("user", &input).await;
                (id, hist, rag)
            } else {
                (std::collections::HashMap::new(), String::new(), Vec::new())
            };

            // 4. Montar prompt com template e formato do modelo selecionado
            let prompt_chatml = PromptComposer::compose_with_template(
                &identity,
                &emotion_now,
                &rag_snippets,
                &history_chatml,
                &input,
                &lang_now,
                Some(&prompt_tpl),
                &prompt_preset,
            );

            // 5. Inferência em spawn_blocking com streaming `_TKN_:` + buffer TTS ignorando `<think>`
            let brain_clone = Arc::clone(&brain);
            let ui_tx_stream = ui_tx.clone();
            let speech_stream = speech.clone();
            let input_for_brain = input.clone();
            let rag_for_brain = rag_snippets.clone();

            let gen_result = tokio::task::spawn_blocking(move || {
                let mut guard = brain_clone.lock();
                let mut in_think = false;
                let mut tts_buffer = String::new();

                let res = guard.generate_stream(
                    &prompt_chatml,
                    &input_for_brain,
                    &emotion_now,
                    &rag_for_brain,
                    llm_temp,
                    &sampler_cfg,
                    |token| {
                        let _ = ui_tx_stream.send(format!("_TKN_:{}", token));

                        // Bufferização de sentenças para TTS ignorando blocos <think>...</think>
                        if token.contains("<think>") {
                            in_think = true;
                        }
                        if in_think {
                            if let Some(after) = token.split("</think>").nth(1) {
                                in_think = false;
                                tts_buffer.push_str(after);
                            }
                            return;
                        }

                        tts_buffer.push_str(token);
                        if tts_buffer.ends_with('.')
                            || tts_buffer.ends_with('!')
                            || tts_buffer.ends_with('?')
                            || tts_buffer.ends_with('\n')
                        {
                            let sentence = sanitize_assistant_text(&tts_buffer, false);
                            if !sentence.is_empty() {
                                speech_stream.speak(sentence);
                            }
                            tts_buffer.clear();
                        }
                    },
                );

                if !tts_buffer.trim().is_empty() && !in_think {
                    let sentence = sanitize_assistant_text(&tts_buffer, false);
                    if !sentence.is_empty() {
                        speech_stream.speak(sentence);
                    }
                }

                res
            })
            .await;

            let _ = ui_tx.send("_STREAM_END_".to_string());

            // 6. Pós-processamento assíncrono + garantia de reset de flags mesmo em caso de erro
            match gen_result {
                Ok(Ok(full_raw)) => {
                    let clean_reply = sanitize_assistant_text(&full_raw, false);
                    if let Some(ref mem) = memory {
                        if !clean_reply.is_empty() {
                            let _ = mem.add_message("assistant", &clean_reply).await;
                        }
                    }

                    // Extração de fatos assíncrona com a Amígdala
                    let am_clone = Arc::clone(&amygdala);
                    let shared_post = Arc::clone(&shared);
                    let mem_post = memory.clone();
                    let mem_snap_post = Arc::clone(&mem_snap);
                    let ui_tx_post = ui_tx.clone();
                    let input_post = input.clone();
                    let raw_post = full_raw.clone();

                    tokio::spawn(async move {
                        {
                            shared_post.write().is_amygdala_thinking = true;
                        }
                        let extracted = tokio::task::spawn_blocking(move || {
                            am_clone.lock().extract_facts(&input_post, &raw_post)
                        })
                        .await
                        .unwrap_or_default();

                        if let Some(ref mem) = mem_post {
                            for fact in extracted {
                                let old_name = shared_post.read().assistant_name.clone();
                                if let Err(e) = mem
                                    .learn_fact(
                                        &fact.subject,
                                        &fact.attribute,
                                        &fact.value,
                                        fact.relevance,
                                    )
                                    .await
                                {
                                    EvolveMonitor::global().record_event(
                                        HealthLevel::Error,
                                        "memory",
                                        format!("Erro ao persistir fato: {}", e),
                                    );
                                } else {
                                    EvolveMonitor::global().record_event(
                                        HealthLevel::Info,
                                        "memory",
                                        format!(
                                            "[LEARN] {} | {} | {} (relevância: {:.2})",
                                            fact.subject, fact.attribute, fact.value, fact.relevance
                                        ),
                                    );
                                    if let Ok(n) = sync_assistant_name_if_changed!(
                                        mem,
                                        &fact.subject,
                                        &fact.attribute,
                                        &old_name,
                                        &fact.value,
                                        100
                                    ) {
                                        if is_assistant_name_fact(&fact.subject, &fact.attribute) {
                                            shared_post.write().assistant_name = fact.value.trim().to_string();
                                            let _ = ui_tx_post.send(format!("_RENAME_:{}|{}", old_name, fact.value.trim()));
                                            if n > 0 {
                                                EvolveMonitor::global().record_event(
                                                    HealthLevel::Info,
                                                    "memory",
                                                    format!("[NAME_SYNC] {} referências recentes atualizadas ({} -> {})", n, old_name, fact.value.trim()),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        sync_memory_snapshot(&mem_post, &mem_snap_post).await;

                        // Sempre resetar flag mesmo se houver erro
                        shared_post.write().is_amygdala_thinking = false;
                    });
                }
                Ok(Err(e)) => {
                    EvolveMonitor::global().record_event(
                        HealthLevel::Error,
                        "brain",
                        format!("Erro na inferência: {}", e),
                    );
                }
                Err(e) => {
                    EvolveMonitor::global().record_event(
                        HealthLevel::Error,
                        "core",
                        format!("Falha na task de inferência: {}", e),
                    );
                }
            }

            {
                let mut st = shared.write();
                st.is_brain_thinking = false;
                st.is_amygdala_thinking = false;
            }
        }
    });
}

/// Rotina "The Blind Vault" (Modo Sonho):
/// Lê `Diag.json`, traduz erros técnicos em descrições legíveis via LLM de forma isolada
/// (sem poluir `chat_history`) e grava `DreamReport.md`.
async fn run_blind_vault_dream(
    brain: Arc<Mutex<Brain>>,
    shared: SharedStateHandle,
    force_even_if_empty: bool,
) {
    let diag_summary = EvolveMonitor::global().read_diag_errors();
    let input_summary = match diag_summary {
        Some(s) => s,
        None if force_even_if_empty => {
            "Nenhum erro registrado em Diag.json até o momento. Todas as operações estão saudáveis."
                .to_string()
        }
        None => return,
    };

    let isolated_prompt = format!(
        "<|im_start|>system\n\
         Você é um engenheiro de diagnóstico de sistemas. Traduza os erros técnicos abaixo para um resumo claro em PT-BR com possíveis causas e ações recomendadas.\n\
         <|im_end|>\n\
         <|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
        input_summary
    );

    let emotion = shared.read().emotion;
    let report = tokio::task::spawn_blocking(move || {
        let mut guard = brain.lock();
        guard.generate_stream(
            &isolated_prompt,
            &input_summary,
            &emotion,
            &[],
            0.3,
            &SamplerConfig::default(),
            |_| {},
        )
    })
    .await;

    if let Ok(Ok(raw)) = report {
        let clean = sanitize_assistant_text(&raw, false);
        if !clean.is_empty() {
            if let Ok(path) = EvolveMonitor::global().write_dream_report(&clean) {
                EvolveMonitor::global().record_event(
                    HealthLevel::Info,
                    "evolve",
                    format!("[BLIND_VAULT] Relatório Modo Sonho salvo em {:?}", path),
                );
            }
            shared.write().last_dream_report = Some(clean);
        }
    }
}

/// Loop de manutenção: executa `log_health()` e `apply_homeostasis()` a cada 10s (ou 30s em `power_save`).
fn spawn_maintenance_loop(rt: &tokio::runtime::Handle, shared: SharedStateHandle) {
    rt.spawn(async move {
        loop {
            let interval_secs = if shared.read().power_save { 30 } else { 10 };
            sleep(Duration::from_secs(interval_secs)).await;
            let snap = EvolveMonitor::global().log_health();
            shared
                .write()
                .emotion
                .apply_homeostasis(snap.cpu_usage_percent);
        }
    });
}

fn main() -> eframe::Result<()> {
    EvolveMonitor::init_tracing();
    EvolveMonitor::global().set_diag_path(executable_dir().join("Diag.json"));
    let _boot_timer = PerfTimer::start("core_boot");
    EvolveMonitor::global().log_health();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Falha ao iniciar Tokio runtime");

    let shared = new_shared_state();
    let modelos_dir = executable_modelos_dir();
    let saved_settings = AppSettings::load_default_or_init();
    {
        let mut st = shared.write();
        saved_settings.apply_to_shared(&mut st);
        rust_i18n::set_locale(&st.language);
    }

    let (speech_handle, speech_cfg) =
        SpeechEngine::spawn(Some(resolve_exe_or_cwd_file("speech.json")));
    if !AppSettings::default_path().exists() {
        let mut st = shared.write();
        st.tts_enabled = speech_cfg.enabled;
        st.tts_model_path = speech_cfg.model_path;
    } else {
        speech_handle.set_enabled(saved_settings.tts_enabled);
        speech_handle.load_model(&saved_settings.tts_model_path);
    }

    // Prioriza modelo salvo em settings.json se existir em disco; caso contrário, usa o primeiro .gguf em modelos/
    let initial_llm_path = if std::path::Path::new(&saved_settings.llm_model_path).is_file() {
        PathBuf::from(&saved_settings.llm_model_path)
    } else {
        discover_gguf_models(&modelos_dir)
            .into_iter()
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&saved_settings.llm_model_path))
    };
    // Prioriza SLM salvo em settings.json se existir em disco; caso contrário, escolhe a melhor SLM .gguf em modelos/
    let initial_slm_path = if std::path::Path::new(&saved_settings.slm_model_path).is_file() {
        PathBuf::from(&saved_settings.slm_model_path)
    } else {
        let mut slm_candidates: Vec<String> = discover_amygdala_models(&modelos_dir)
            .into_iter()
            .filter(|p| p.to_lowercase().ends_with(".gguf"))
            .collect();
        slm_candidates.sort_by_key(|p| {
            let lower = p.to_lowercase();
            let same_as_llm = std::path::Path::new(p) == initial_llm_path.as_path();
            let is_ptbr_tiny = lower.contains("ptbr") || lower.contains("tinyllama");
            let is_smol = lower.contains("smollm");
            (same_as_llm, !is_ptbr_tiny, !is_smol, p.len())
        });
        slm_candidates
            .into_iter()
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(&saved_settings.slm_model_path))
    };

    let initial_ctx_size = {
        let mut st = shared.write();
        st.llm_model_path = initial_llm_path.display().to_string();
        st.slm_model_path = initial_slm_path.display().to_string();
        st.llm_ctx_size
    };
    refresh_available_models(&shared);

    let brain = Arc::new(Mutex::new(Brain::new(
        Some(initial_llm_path),
        initial_ctx_size,
    )));
    let amygdala = Arc::new(Mutex::new(Amygdala::new(Some(initial_slm_path))));

    let convo_test = std::env::args().any(|a| a == "--convo-test");

    let memory = rt.block_on(async {
        let mem_res = if convo_test {
            Memory::open_in_memory().await
        } else {
            Memory::open_default().await
        };
        match mem_res {
            Ok(mem) => {
                let _ = mem.learn_fact("Assistente", "nome", "Theia", 1.0).await;
                let _ = mem.learn_fact("Assistente", "idioma", "PT-BR", 1.0).await;
                let _ = mem
                    .learn_fact("Assistente", "papel", "Assistente Virtual", 1.0)
                    .await;
                if let Ok(purged) = mem.purge_stale_prototype_messages().await {
                    if purged > 0 {
                        EvolveMonitor::global().record_event(
                            HealthLevel::Info,
                            "memory",
                            format!(
                                "Removidas {} mensagens antigas de fallback do histórico.",
                                purged
                            ),
                        );
                    }
                }
                Some(mem)
            }
            Err(e) => {
                EvolveMonitor::global().record_event(
                    HealthLevel::Error,
                    "memory",
                    format!("Falha ao abrir SQLite: {}", e),
                );
                None
            }
        }
    });

    let mem_snap: MemorySnapshotHandle = Arc::new(RwLock::new(MemorySnapshot::default()));
    rt.block_on(sync_memory_snapshot(&memory, &mem_snap));

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<CoreCommand>();
    let (ui_tx, ui_rx) = mpsc::unbounded_channel::<String>();

    spawn_core_processor(
        rt.handle(),
        cmd_rx,
        ui_tx,
        Arc::clone(&shared),
        memory,
        Arc::clone(&mem_snap),
        Arc::clone(&brain),
        amygdala,
        speech_handle.clone(),
    );
    spawn_maintenance_loop(rt.handle(), Arc::clone(&shared));

    if !convo_test {
        let brain_boot = Arc::clone(&brain);
        let shared_boot = Arc::clone(&shared);
        rt.spawn(async move {
            run_blind_vault_dream(brain_boot, shared_boot, false).await;
        });
    }

    if convo_test {
        let exit_code = rt.block_on(run_convo_test(cmd_tx, ui_rx, mem_snap));
        std::process::exit(exit_code);
    }

    let smoke_test = std::env::args().any(|a| a == "--smoke-test");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_title(format!("ThrediaRs v{}", env!("CARGO_PKG_VERSION"))),
        ..Default::default()
    };

    let _rt_guard = rt;
    eframe::run_native(
        "ThrediaRs",
        options,
        Box::new(move |cc| {
            Box::new(ThrediaApp::new(
                cc,
                smoke_test,
                shared,
                mem_snap,
                cmd_tx,
                ui_rx,
                speech_handle,
            ))
        }),
    )
}

async fn run_convo_test(
    cmd_tx: UnboundedSender<CoreCommand>,
    mut ui_rx: UnboundedReceiver<String>,
    mem_snap: MemorySnapshotHandle,
) -> i32 {
    let turns = [
        "limpar",
        "Oi! Meu nome é Axel e eu gosto de programar em Rust.",
        "Qual é o meu nome e o que eu gosto de fazer?",
        "status",
        "debug mem",
    ];

    println!("=== INICIANDO TESTE CONVERSACIONAL END-TO-END ===");
    let mut failed = false;

    for (idx, prompt) in turns.iter().enumerate() {
        let t0 = std::time::Instant::now();
        println!("\n[Turno {}] Você: {}", idx + 1, prompt);
        if cmd_tx
            .send(CoreCommand::UserMessage((*prompt).to_string()))
            .is_err()
        {
            eprintln!("ERRO: falha ao enviar comando para o córtex");
            return 1;
        }

        let mut raw = String::new();
        let timeout = tokio::time::timeout(Duration::from_secs(60), async {
            while let Some(msg) = ui_rx.recv().await {
                if let Some(tok) = msg.strip_prefix("_TKN_:") {
                    raw.push_str(tok);
                } else if msg == "_STREAM_END_" {
                    break;
                }
            }
        })
        .await;

        if timeout.is_err() {
            eprintln!("ERRO: timeout no turno {}", idx + 1);
            return 1;
        }

        let clean = sanitize_assistant_text(&raw, false);
        let elapsed = t0.elapsed().as_millis();
        println!("[Turno {}] Theia ({} ms): {}", idx + 1, elapsed, clean);

        if clean.is_empty() {
            eprintln!("FALHA: resposta vazia no turno {}", idx + 1);
            failed = true;
        }
        if clean.contains("<|im_end|>") || clean.contains("<|im_start|>") {
            eprintln!("FALHA: vazamento de token ChatML no turno {}", idx + 1);
            failed = true;
        }
        // Aguarda pós-processamento assíncrono entre turnos para o RAG refletir fatos recém-aprendidos
        sleep(Duration::from_millis(150)).await;
    }

    // Aguarda pós-processamento assíncrono de fatos
    sleep(Duration::from_millis(250)).await;
    let _ = cmd_tx.send(CoreCommand::RefreshMemoryView);
    sleep(Duration::from_millis(100)).await;

    let facts = mem_snap.read().facts.clone();
    println!("\n[Fatos em Memória ({})]:", facts.len());
    for f in &facts {
        println!("  - [{}] {} = {}", f.subject, f.attribute, f.value);
    }

    if !facts
        .iter()
        .any(|f| f.subject == "Usuario" && f.attribute == "nome" && f.value == "Axel")
    {
        eprintln!("FALHA: fato `Usuario | nome = Axel` não foi persistido!");
        failed = true;
    }

    if failed {
        println!("\n=== TESTE CONVERSACIONAL FALHOU ===");
        1
    } else {
        println!("\n=== TESTE CONVERSACIONAL PASSOU ===");
        0
    }
}
