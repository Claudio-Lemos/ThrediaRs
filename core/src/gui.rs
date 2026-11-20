use crate::commands::{sanitize_assistant_text, CoreCommand, MemorySnapshotHandle};
use crate::settings::AppSettings;
use brain::{
    builtin_prompt_presets, find_builtin_preset_for_prompt, ComputeBackend, CustomPromptPreset,
    EmotionState, SamplerConfig, SharedStateHandle,
};
use eframe::egui::{self, Color32, RichText, Rounding, Stroke, Vec2};
use evolve::{EvolveMonitor, HealthLevel};
use speech::SpeechHandle;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidebarTab {
    Cerebro,
    Memoria,
    Sistema,
    Voz,
    Logs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryViewMode {
    Fatos,
    Historico,
}

#[derive(Debug, Clone)]
struct ChatBubble {
    is_user: bool,
    sender: String,
    text: String,
}

#[derive(Debug, Clone)]
enum FilePickResult {
    LlmGguf(PathBuf),
    SlmModel(PathBuf),
    VoiceOnnx(PathBuf),
}

pub struct ThrediaApp {
    smoke_test: bool,
    frames: u32,
    shared: SharedStateHandle,
    mem_snap: MemorySnapshotHandle,
    cmd_tx: UnboundedSender<CoreCommand>,
    ui_rx: UnboundedReceiver<String>,
    speech: SpeechHandle,
    active_tab: SidebarTab,
    input_buf: String,
    messages: Vec<ChatBubble>,
    streaming_raw: String,
    is_streaming: bool,
    log_search: String,
    selected_llm_path: String,
    selected_slm_path: String,
    selected_backend: ComputeBackend,
    move_to_modelos: bool,
    default_dialog_dir: PathBuf,
    file_pick_tx: Sender<FilePickResult>,
    file_pick_rx: Receiver<FilePickResult>,
    mem_view_mode: MemoryViewMode,
    mem_search: String,
    selected_fact_subject: Option<String>,
    new_fact_subject: String,
    new_fact_attr: String,
    new_fact_value: String,
    new_fact_conf: f64,
    editing_fact_id: Option<i64>,
    edit_fact_subject: String,
    edit_fact_attr: String,
    edit_fact_value: String,
    edit_fact_conf: f64,
    custom_prompt_name: String,
    last_saved_settings_json: String,
}

impl ThrediaApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        smoke_test: bool,
        shared: SharedStateHandle,
        mem_snap: MemorySnapshotHandle,
        cmd_tx: UnboundedSender<CoreCommand>,
        ui_rx: UnboundedReceiver<String>,
        speech: SpeechHandle,
    ) -> Self {
        apply_dark_messenger_theme(&cc.egui_ctx);
        let (
            assistant_name,
            selected_llm_path,
            selected_slm_path,
            selected_backend,
            move_to_modelos,
            initial_json,
        ) = {
            let st = shared.read();
            let json = serde_json::to_string(&AppSettings::from_shared(&st)).unwrap_or_default();
            (
                st.assistant_name.clone(),
                st.llm_model_path.clone(),
                st.slm_model_path.clone(),
                st.compute_backend,
                st.move_to_modelos,
                json,
            )
        };
        let home_dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("/"));
        let (file_pick_tx, file_pick_rx) = mpsc::channel();

        Self {
            smoke_test,
            frames: 0,
            shared,
            mem_snap,
            cmd_tx,
            ui_rx,
            speech,
            active_tab: SidebarTab::Cerebro,
            input_buf: String::new(),
            messages: vec![ChatBubble {
                is_user: false,
                sender: assistant_name.clone(),
                text: t!("greeting", name = assistant_name).to_string(),
            }],
            streaming_raw: String::new(),
            is_streaming: false,
            log_search: String::new(),
            selected_llm_path,
            selected_slm_path,
            selected_backend,
            move_to_modelos,
            default_dialog_dir: home_dir,
            file_pick_tx,
            file_pick_rx,
            mem_view_mode: MemoryViewMode::Fatos,
            mem_search: String::new(),
            selected_fact_subject: None,
            new_fact_subject: "Usuario".to_string(),
            new_fact_attr: String::new(),
            new_fact_value: String::new(),
            new_fact_conf: 1.0,
            editing_fact_id: None,
            edit_fact_subject: String::new(),
            edit_fact_attr: String::new(),
            edit_fact_value: String::new(),
            edit_fact_conf: 1.0,
            custom_prompt_name: String::new(),
            last_saved_settings_json: initial_json,
        }
    }

    fn spawn_native_gguf_picker(&self) {
        let tx = self.file_pick_tx.clone();
        let start_dir = self.default_dialog_dir.clone();
        let title = t!("dialog.select_gguf_title").to_string();
        let filter = t!("dialog.select_gguf_filter").to_string();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_title(&title)
                .set_directory(&start_dir)
                .add_filter(&filter, &["gguf"])
                .pick_file()
            {
                let _ = tx.send(FilePickResult::LlmGguf(path));
            }
        });
    }

    fn spawn_native_onnx_picker(&self) {
        let tx = self.file_pick_tx.clone();
        let start_dir = self.default_dialog_dir.clone();
        let title = t!("dialog.select_onnx_title").to_string();
        let filter = t!("dialog.select_onnx_filter").to_string();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_title(&title)
                .set_directory(&start_dir)
                .add_filter(&filter, &["onnx"])
                .pick_file()
            {
                let _ = tx.send(FilePickResult::VoiceOnnx(path));
            }
        });
    }

    fn apply_selected_gguf(&mut self, path: &Path) {
        let p_str = path.display().to_string();
        let ctx_size_now = self.shared.read().llm_ctx_size;
        self.selected_llm_path = p_str.clone();
        if let Some(parent) = path.parent() {
            self.default_dialog_dir = parent.to_path_buf();
        }
        let _ = self.cmd_tx.send(CoreCommand::HotSwapLlm {
            model_path: p_str,
            n_ctx: ctx_size_now,
            backend: self.selected_backend,
            move_to_modelos: self.move_to_modelos,
        });
    }

    fn spawn_native_slm_picker(&self) {
        let tx = self.file_pick_tx.clone();
        let start_dir = self.default_dialog_dir.clone();
        let title = t!("dialog.select_slm_title").to_string();
        let filter = t!("dialog.select_slm_filter").to_string();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_title(&title)
                .set_directory(&start_dir)
                .add_filter(&filter, &["gguf", "onnx"])
                .pick_file()
            {
                let _ = tx.send(FilePickResult::SlmModel(path));
            }
        });
    }

    fn apply_selected_slm(&mut self, path: &Path) {
        let p_str = path.display().to_string();
        self.selected_slm_path = p_str.clone();
        if let Some(parent) = path.parent() {
            self.default_dialog_dir = parent.to_path_buf();
        }
        let _ = self.cmd_tx.send(CoreCommand::HotSwapSlm {
            model_path: p_str,
            move_to_modelos: self.move_to_modelos,
        });
    }

    fn auto_save_settings_if_changed(&mut self) {
        if self.frames % 15 != 0 {
            return;
        }
        let current_json = {
            let st = self.shared.read();
            serde_json::to_string(&AppSettings::from_shared(&st)).unwrap_or_default()
        };
        if !current_json.is_empty() && current_json != self.last_saved_settings_json {
            self.last_saved_settings_json = current_json;
            let _ = self.cmd_tx.send(CoreCommand::SaveSettings);
        }
    }

    fn drain_file_picks(&mut self) {
        while let Ok(picked) = self.file_pick_rx.try_recv() {
            match picked {
                FilePickResult::LlmGguf(path) => {
                    self.apply_selected_gguf(&path);
                }
                FilePickResult::SlmModel(path) => {
                    self.apply_selected_slm(&path);
                }
                FilePickResult::VoiceOnnx(path) => {
                    let p_str = path.display().to_string();
                    self.shared.write().tts_model_path = p_str.clone();
                    self.speech.load_model(p_str);
                }
            }
        }
    }

    fn drain_stream(&mut self) {
        while let Ok(msg) = self.ui_rx.try_recv() {
            if let Some(tok) = msg.strip_prefix("_TKN_:") {
                if !self.is_streaming {
                    self.is_streaming = true;
                    self.streaming_raw.clear();
                }
                self.streaming_raw.push_str(tok);
            } else if msg == "_STREAM_END_" {
                let (keep, name) = {
                    let st = self.shared.read();
                    (st.show_thinking, st.assistant_name.clone())
                };
                let final_text = sanitize_assistant_text(&self.streaming_raw, keep);
                if !final_text.is_empty() {
                    self.messages.push(ChatBubble {
                        is_user: false,
                        sender: name,
                        text: final_text,
                    });
                }
                self.streaming_raw.clear();
                self.is_streaming = false;
            } else if let Some(sys_msg) = msg.strip_prefix("_SYS_:") {
                self.messages.push(ChatBubble {
                    is_user: false,
                    sender: t!("sender.system").to_string(),
                    text: sys_msg.to_string(),
                });
            } else if let Some(rename_payload) = msg.strip_prefix("_RENAME_:") {
                if let Some((old_name, new_name)) = rename_payload.split_once('|') {
                    for b in &mut self.messages {
                        if !b.is_user && b.sender.eq_ignore_ascii_case(old_name) {
                            b.sender = new_name.to_string();
                        }
                        if !old_name.is_empty() && b.text.contains(old_name) {
                            b.text = b.text.replace(old_name, new_name);
                        }
                    }
                }
            }
        }
    }

    fn mood_color(emotion: &EmotionState) -> Color32 {
        if emotion.valence >= 0.35 {
            Color32::from_rgb(76, 217, 100) // Verde positivo
        } else if emotion.valence <= -0.35 {
            Color32::from_rgb(255, 89, 94) // Vermelho tenso/negativo
        } else if emotion.arousal >= 0.60 {
            Color32::from_rgb(255, 190, 59) // Âmbar alerta
        } else {
            Color32::from_rgb(100, 181, 246) // Azul calmo/equilibrado
        }
    }

    fn send_current_input(&mut self) {
        let trimmed = self.input_buf.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        self.input_buf.clear();
        self.messages.push(ChatBubble {
            is_user: true,
            sender: t!("sender.user").to_string(),
            text: trimmed.clone(),
        });
        let _ = self.cmd_tx.send(CoreCommand::UserMessage(trimmed));
    }

    fn render_header(&self, ui: &mut egui::Ui, state: &brain::SharedState) {
        let mood_color = Self::mood_color(&state.emotion);
        let thinking = state.is_brain_thinking || state.is_amygdala_thinking || self.is_streaming;

        ui.horizontal(|ui| {
            // Avatar circular desenhado via painter
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(38.0), egui::Sense::hover());
            let center = rect.center();
            ui.painter().circle_filled(center, 18.0, Color32::from_rgb(45, 52, 70));
            ui.painter()
                .circle_stroke(center, 18.0, Stroke::new(2.0_f32, mood_color));
            let initial = state
                .assistant_name
                .chars()
                .next()
                .unwrap_or('A')
                .to_uppercase()
                .to_string();
            ui.painter().text(
                center,
                egui::Align2::CENTER_CENTER,
                initial,
                egui::FontId::proportional(18.0),
                Color32::WHITE,
            );

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(&state.assistant_name)
                            .strong()
                            .size(16.0)
                            .color(Color32::WHITE),
                    );
                    ui.label(
                        RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .small()
                            .color(Color32::GRAY),
                    );
                });
                ui.horizontal(|ui| {
                    let (status_txt, status_col) = if state.is_swapping_model {
                        (t!("status.swapping"), Color32::from_rgb(130, 190, 255))
                    } else if thinking {
                        (t!("status.thinking"), Color32::from_rgb(255, 190, 59))
                    } else {
                        (t!("status.online"), Color32::from_rgb(76, 217, 100))
                    };
                    ui.label(RichText::new(status_txt.as_ref()).small().color(status_col));
                    ui.label(RichText::new("|").small().color(Color32::DARK_GRAY));
                    ui.label(
                        RichText::new(state.emotion.get_mood_desc_for(&state.language))
                            .small()
                            .strong()
                            .color(mood_color),
                    );
                });
            });
        });
    }

    fn render_chat_area(&self, ui: &mut egui::Ui, show_thinking: bool) {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(6.0);
                for bubble in &self.messages {
                    render_bubble(ui, bubble.is_user, &bubble.sender, &bubble.text);
                    ui.add_space(6.0);
                }

                if self.is_streaming {
                    let assistant_name = self.shared.read().assistant_name.clone();
                    let preview = sanitize_assistant_text(&self.streaming_raw, show_thinking);
                    let display = if preview.is_empty() {
                        t!("status.thinking_short").to_string()
                    } else {
                        preview
                    };
                    let typing_label = t!("status.typing", name = assistant_name);
                    render_bubble(ui, false, typing_label.as_ref(), &display);
                }
                ui.add_space(8.0);
            });
    }

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.active_tab, SidebarTab::Cerebro, t!("tab.brain").as_ref());
            ui.selectable_value(&mut self.active_tab, SidebarTab::Memoria, t!("tab.memory").as_ref());
            ui.selectable_value(&mut self.active_tab, SidebarTab::Sistema, t!("tab.system").as_ref());
            ui.selectable_value(&mut self.active_tab, SidebarTab::Voz, t!("tab.voice").as_ref());
            ui.selectable_value(&mut self.active_tab, SidebarTab::Logs, t!("tab.logs").as_ref());
        });
        ui.separator();

        match self.active_tab {
            SidebarTab::Cerebro => self.render_tab_cerebro(ui),
            SidebarTab::Memoria => self.render_tab_memoria(ui),
            SidebarTab::Sistema => self.render_tab_sistema(ui),
            SidebarTab::Voz => self.render_tab_voz(ui),
            SidebarTab::Logs => self.render_tab_logs(ui),
        }
    }

    fn render_tab_memoria(&mut self, ui: &mut egui::Ui) {
        let snap = self.mem_snap.read().clone();

        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.mem_view_mode,
                MemoryViewMode::Fatos,
                t!("memory.facts_count", count = snap.facts.len()).as_ref(),
            );
            ui.selectable_value(
                &mut self.mem_view_mode,
                MemoryViewMode::Historico,
                t!("memory.history_count", count = snap.messages.len()).as_ref(),
            );
            if ui
                .button("↻")
                .on_hover_text(t!("memory.reload_tooltip").as_ref())
                .clicked()
            {
                let _ = self.cmd_tx.send(CoreCommand::RefreshMemoryView);
            }
            if ui
                .button(t!("memory.open_sql").as_ref())
                .on_hover_text(t!("memory.open_sql_tooltip").as_ref())
                .clicked()
            {
                open_memory_sql_file();
            }
        });

        ui.horizontal(|ui| {
            ui.label(t!("memory.filter").as_ref());
            ui.text_edit_singleline(&mut self.mem_search);
        });
        ui.separator();

        let query = self.mem_search.to_lowercase();

        match self.mem_view_mode {
            MemoryViewMode::Fatos => {
                egui::CollapsingHeader::new(t!("memory.add_fact").as_ref())
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(t!("memory.subject").as_ref());
                            ui.text_edit_singleline(&mut self.new_fact_subject);
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("memory.attribute").as_ref());
                            ui.text_edit_singleline(&mut self.new_fact_attr);
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("memory.value").as_ref());
                            ui.text_edit_singleline(&mut self.new_fact_value);
                        });
                        ui.add(
                            egui::Slider::new(&mut self.new_fact_conf, 0.0..=1.0)
                                .text(t!("memory.confidence").as_ref())
                                .fixed_decimals(2),
                        );
                        if ui.button(t!("memory.save_fact").as_ref()).clicked()
                            && !self.new_fact_subject.trim().is_empty()
                            && !self.new_fact_attr.trim().is_empty()
                            && !self.new_fact_value.trim().is_empty()
                        {
                            let _ = self.cmd_tx.send(CoreCommand::AddFact {
                                subject: self.new_fact_subject.trim().to_string(),
                                attribute: self.new_fact_attr.trim().to_string(),
                                value: self.new_fact_value.trim().to_string(),
                                confidence: self.new_fact_conf,
                            });
                            self.new_fact_attr.clear();
                            self.new_fact_value.clear();
                        }
                    });

                ui.add_space(6.0);

                // Agrupa sujeitos como categorias em grade 3x3 scrollável
                let mut categories: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                for fact in &snap.facts {
                    *categories.entry(fact.subject.clone()).or_insert(0) += 1;
                }

                if !categories.is_empty() {
                    let header_label = match &self.selected_fact_subject {
                        Some(sel) => format!("{} [{}]", t!("memory.categories_title"), sel),
                        None => t!("memory.categories_title").to_string(),
                    };

                    egui::CollapsingHeader::new(header_label)
                        .id_source("memory_categories_collapse")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let all_selected = self.selected_fact_subject.is_none();
                                if ui
                                    .selectable_label(
                                        all_selected,
                                        t!("memory.all_categories", count = snap.facts.len())
                                            .as_ref(),
                                    )
                                    .clicked()
                                {
                                    self.selected_fact_subject = None;
                                }
                            });

                            let cat_list: Vec<(String, usize)> = categories.into_iter().collect();
                            let cell_w = ((ui.available_width() - 16.0) / 3.0).max(70.0);
                            let cell_h = 34.0;

                            egui::ScrollArea::vertical()
                                .id_source("memory_subject_grid_scroll")
                                .max_height(120.0)
                                .show(ui, |ui| {
                                    for row in cat_list.chunks(3) {
                                        ui.horizontal(|ui| {
                                            for (subj, count) in row {
                                                let is_sel = self
                                                    .selected_fact_subject
                                                    .as_ref()
                                                    .is_some_and(|s| s == subj);
                                                let fill = if is_sel {
                                                    Color32::from_rgb(48, 92, 160)
                                                } else {
                                                    Color32::from_rgb(36, 40, 52)
                                                };
                                                let stroke = if is_sel {
                                                    Stroke::new(
                                                        1.5_f32,
                                                        Color32::from_rgb(130, 190, 255),
                                                    )
                                                } else {
                                                    Stroke::new(
                                                        1.0_f32,
                                                        Color32::from_rgb(56, 62, 78),
                                                    )
                                                };
                                                let btn = egui::Button::new(
                                                    RichText::new(format!("{} ({})", subj, count))
                                                        .small()
                                                        .color(Color32::WHITE),
                                                )
                                                .fill(fill)
                                                .stroke(stroke)
                                                .rounding(Rounding::same(6.0))
                                                .min_size(Vec2::new(cell_w, cell_h));

                                                if ui.add_sized([cell_w, cell_h], btn).clicked() {
                                                    if is_sel {
                                                        self.selected_fact_subject = None;
                                                    } else {
                                                        self.selected_fact_subject =
                                                            Some(subj.clone());
                                                    }
                                                }
                                            }
                                        });
                                        ui.add_space(4.0);
                                    }
                                });
                        });
                    ui.separator();
                }

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for fact in &snap.facts {
                        if let Some(ref sel_subj) = self.selected_fact_subject {
                            if &fact.subject != sel_subj {
                                continue;
                            }
                        }
                        if !query.is_empty()
                            && !fact.subject.to_lowercase().contains(&query)
                            && !fact.attribute.to_lowercase().contains(&query)
                            && !fact.value.to_lowercase().contains(&query)
                        {
                            continue;
                        }

                        egui::Frame::none()
                            .fill(Color32::from_rgb(32, 36, 46))
                            .rounding(Rounding::same(6.0))
                            .inner_margin(egui::Margin::same(8.0))
                            .show(ui, |ui| {
                                if self.editing_fact_id == Some(fact.id) {
                                    ui.label(
                                        RichText::new(t!("memory.editing_fact", id = fact.id).as_ref())
                                            .small()
                                            .strong(),
                                    );
                                    ui.horizontal(|ui| {
                                        ui.label(t!("memory.subject").as_ref());
                                        ui.text_edit_singleline(&mut self.edit_fact_subject);
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label(t!("memory.attribute").as_ref());
                                        ui.text_edit_singleline(&mut self.edit_fact_attr);
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label(t!("memory.value").as_ref());
                                        ui.text_edit_singleline(&mut self.edit_fact_value);
                                    });
                                    ui.add(
                                        egui::Slider::new(&mut self.edit_fact_conf, 0.0..=1.0)
                                            .text(t!("memory.conf_short").as_ref())
                                            .fixed_decimals(2),
                                    );
                                    ui.horizontal(|ui| {
                                        if ui.button(t!("memory.save").as_ref()).clicked() {
                                            let _ = self.cmd_tx.send(CoreCommand::UpdateFact {
                                                id: fact.id,
                                                subject: self.edit_fact_subject.clone(),
                                                attribute: self.edit_fact_attr.clone(),
                                                value: self.edit_fact_value.clone(),
                                                confidence: self.edit_fact_conf,
                                            });
                                            self.editing_fact_id = None;
                                        }
                                        if ui.button(t!("memory.cancel").as_ref()).clicked() {
                                            self.editing_fact_id = None;
                                        }
                                    });
                                } else {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!("#{} [{}]", fact.id, fact.subject))
                                                .small()
                                                .strong()
                                                .color(Color32::from_rgb(130, 190, 255)),
                                        );
                                        ui.label(
                                            RichText::new(format!("conf: {:.2}", fact.confidence))
                                                .small()
                                                .color(Color32::GRAY),
                                        );
                                    });
                                    ui.label(format!("{} = {}", fact.attribute, fact.value));
                                    ui.horizontal(|ui| {
                                        if ui.small_button(t!("memory.edit").as_ref()).clicked() {
                                            self.editing_fact_id = Some(fact.id);
                                            self.edit_fact_subject = fact.subject.clone();
                                            self.edit_fact_attr = fact.attribute.clone();
                                            self.edit_fact_value = fact.value.clone();
                                            self.edit_fact_conf = fact.confidence;
                                        }
                                        if ui.small_button(t!("memory.delete").as_ref()).clicked() {
                                            let _ = self
                                                .cmd_tx
                                                .send(CoreCommand::DeleteFact { id: fact.id });
                                        }
                                    });
                                }
                            });
                        ui.add_space(4.0);
                    }
                });
            }
            MemoryViewMode::Historico => {
                if ui.button(t!("memory.clear_history").as_ref()).clicked() {
                    let _ = self.cmd_tx.send(CoreCommand::ClearChatHistory);
                }
                ui.add_space(6.0);

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for m in &snap.messages {
                            if !query.is_empty()
                                && !m.role.to_lowercase().contains(&query)
                                && !m.content.to_lowercase().contains(&query)
                            {
                                continue;
                            }
                            egui::Frame::none()
                                .fill(Color32::from_rgb(32, 36, 46))
                                .rounding(Rounding::same(6.0))
                                .inner_margin(egui::Margin::same(8.0))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!("#{} {}", m.id, m.role))
                                                .small()
                                                .strong()
                                                .color(if m.role == "user" {
                                                    Color32::from_rgb(100, 181, 246)
                                                } else {
                                                    Color32::from_rgb(76, 217, 100)
                                                }),
                                        );
                                        ui.label(
                                            RichText::new(&m.created_at)
                                                .small()
                                                .color(Color32::GRAY),
                                        );
                                        if ui.small_button(t!("memory.delete").as_ref()).clicked() {
                                            let _ = self
                                                .cmd_tx
                                                .send(CoreCommand::DeleteChatMessage { id: m.id });
                                        }
                                    });
                                    ui.label(RichText::new(&m.content).small());
                                });
                            ui.add_space(4.0);
                        }
                    });
            }
        }
    }

    fn render_tab_cerebro(&mut self, ui: &mut egui::Ui) {
        let (
            available_models,
            active_model,
            available_slms,
            active_slm,
            is_swapping,
            is_busy,
            ctx_size_now,
        ) = {
            let st = self.shared.read();
            (
                st.available_llm_models.clone(),
                st.llm_model_path.clone(),
                st.available_slm_models.clone(),
                st.slm_model_path.clone(),
                st.is_swapping_model,
                st.is_brain_thinking || st.is_amygdala_thinking,
                st.llm_ctx_size,
            )
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label(RichText::new(t!("brain.llm_section").as_ref()).strong());
        let selected_label = std::path::Path::new(&self.selected_llm_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&self.selected_llm_path)
            .to_string();

        egui::ComboBox::from_id_source("llm_model_combo")
            .selected_text(selected_label)
            .width(240.0)
            .show_ui(ui, |ui| {
                for m in &available_models {
                    let short = std::path::Path::new(m)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(m);
                    ui.selectable_value(&mut self.selected_llm_path, m.clone(), short);
                }
            });

        ui.horizontal(|ui| {
            if ui.button(t!("brain.select_gguf").as_ref()).clicked() {
                self.spawn_native_gguf_picker();
            }
            if ui.button(t!("brain.refresh_list").as_ref()).clicked() {
                let _ = self.cmd_tx.send(CoreCommand::RefreshModels);
            }
        });

        ui.checkbox(
            &mut self.move_to_modelos,
            t!("brain.move_to_modelos").as_ref(),
        );

        ui.horizontal(|ui| {
            ui.label(t!("brain.backend").as_ref());
            ui.selectable_value(&mut self.selected_backend, ComputeBackend::Cpu, "CPU");
            ui.selectable_value(&mut self.selected_backend, ComputeBackend::Gpu, "GPU");
        });

        let swap_enabled = !is_swapping && !is_busy;
        if ui
            .add_enabled(swap_enabled, egui::Button::new(t!("brain.apply_hot_swap").as_ref()))
            .clicked()
        {
            let _ = self.cmd_tx.send(CoreCommand::HotSwapLlm {
                model_path: self.selected_llm_path.clone(),
                n_ctx: ctx_size_now,
                backend: self.selected_backend,
                move_to_modelos: self.move_to_modelos,
            });
        }

        let active_short = std::path::Path::new(&active_model)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&active_model);
        ui.label(
            RichText::new(t!("brain.active_model", name = active_short).as_ref())
                .small()
                .color(Color32::GRAY),
        );

        ui.add_space(8.0);
        ui.separator();

        ui.label(RichText::new(t!("brain.slm_section").as_ref()).strong());
        let selected_slm_label = std::path::Path::new(&self.selected_slm_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&self.selected_slm_path)
            .to_string();

        egui::ComboBox::from_id_source("slm_model_combo")
            .selected_text(selected_slm_label)
            .width(240.0)
            .show_ui(ui, |ui| {
                for m in &available_slms {
                    let short = std::path::Path::new(m)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(m);
                    ui.selectable_value(&mut self.selected_slm_path, m.clone(), short);
                }
            });

        ui.horizontal(|ui| {
            if ui.button(t!("brain.select_slm").as_ref()).clicked() {
                self.spawn_native_slm_picker();
            }
            if ui
                .add_enabled(swap_enabled, egui::Button::new(t!("brain.apply_hot_swap_slm").as_ref()))
                .clicked()
            {
                let _ = self.cmd_tx.send(CoreCommand::HotSwapSlm {
                    model_path: self.selected_slm_path.clone(),
                    move_to_modelos: self.move_to_modelos,
                });
            }
        });

        let active_slm_short = std::path::Path::new(&active_slm)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&active_slm);
        ui.label(
            RichText::new(t!("brain.active_slm", name = active_slm_short).as_ref())
                .small()
                .color(Color32::GRAY),
        );

        ui.add_space(8.0);
        ui.separator();

        let mut st = self.shared.write();

        ui.label(RichText::new(t!("brain.params_section").as_ref()).strong());
        ui.add(
            egui::Slider::new(&mut st.llm_temp, 0.0..=1.5)
                .text(t!("brain.temperature").as_ref())
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.llm_ctx_size, 512..=8192)
                .step_by(512.0)
                .text(t!("brain.ctx_size").as_ref()),
        );
        ui.add(egui::Slider::new(&mut st.sampler.top_k, 1..=120).text("Top-K"));
        ui.add(
            egui::Slider::new(&mut st.sampler.top_p, 0.10..=1.00)
                .text("Top-P")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.min_p, 0.00..=0.50)
                .text("Min-P")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.repeat_penalty, 1.00..=1.50)
                .text("Repeat Penalty")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.repeat_last_n, 16..=256)
                .text("Repeat Last N"),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.freq_penalty, 0.00..=1.00)
                .text("Freq. Penalty")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.presence_penalty, 0.00..=1.00)
                .text("Presence Penalty")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut st.sampler.max_tokens, 32..=1024)
                .step_by(32.0)
                .text("Max Tokens"),
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut st.show_thinking, t!("brain.show_thinking").as_ref());
            if ui.small_button(t!("brain.restore_defaults").as_ref()).clicked() {
                st.llm_temp = 0.7;
                st.sampler = SamplerConfig::default();
            }
        });

        st.move_to_modelos = self.move_to_modelos;

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new(t!("brain.vad_section").as_ref()).strong());

        let mood_col = Self::mood_color(&st.emotion);
        ui.label(
            RichText::new(st.emotion.get_mood_desc_for(&st.language))
                .strong()
                .color(mood_col),
        );

        let val_norm = (st.emotion.valence + 1.0) * 0.5;
        let val_str = format!("{:+.2}", st.emotion.valence);
        let aro_str = format!("{:.2}", st.emotion.arousal);
        let dom_str = format!("{:.2}", st.emotion.dominance);
        ui.add(
            egui::ProgressBar::new(val_norm)
                .text(t!("brain.valence", val = val_str).as_ref()),
        );
        ui.add(
            egui::ProgressBar::new(st.emotion.arousal)
                .text(t!("brain.arousal", val = aro_str).as_ref()),
        );
        ui.add(
            egui::ProgressBar::new(st.emotion.dominance)
                .text(t!("brain.dominance", val = dom_str).as_ref()),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new(t!("brain.amygdala_verdict").as_ref()).strong());
        ui.label(
            RichText::new(&st.last_amygdala_verdict)
                .italics()
                .color(Color32::from_rgb(130, 190, 255)),
        );
        if st.is_amygdala_thinking {
            ui.label(
                RichText::new(t!("brain.amygdala_processing").as_ref())
                    .small()
                    .color(Color32::from_rgb(255, 190, 59)),
            );
        }
        });
    }

    fn render_tab_sistema(&mut self, ui: &mut egui::Ui) {
        let snap = EvolveMonitor::global().snapshot();
        let perf = EvolveMonitor::global().get_performance_summary();
        let samples = EvolveMonitor::global().recent_samples();
        let whisper_acc = EvolveMonitor::global().whisper_accuracy();
        let ml_rate = EvolveMonitor::global().ml_success_rate();
        let dream_report = self.shared.read().last_dream_report.clone();

        egui::ScrollArea::vertical().show(ui, |ui| {
            {
                let mut st = self.shared.write();
                let current_lang_label = if st.language.starts_with("en") {
                    "English"
                } else {
                    "Português (BR)"
                };
                ui.horizontal(|ui| {
                    ui.label(RichText::new(t!("system.language").as_ref()).strong());
                    egui::ComboBox::from_id_source("language_combo")
                        .selected_text(current_lang_label)
                        .width(150.0)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(
                                    &mut st.language,
                                    "pt-BR".to_string(),
                                    "Português (BR)",
                                )
                                .changed()
                            {
                                rust_i18n::set_locale("pt-BR");
                                if let Some(preset) =
                                    find_builtin_preset_for_prompt(&st.system_prompt_template)
                                {
                                    st.system_prompt_template =
                                        preset.prompt_for_lang("pt-BR").to_string();
                                }
                            }
                            if ui
                                .selectable_value(&mut st.language, "en".to_string(), "English")
                                .changed()
                            {
                                rust_i18n::set_locale("en");
                                if let Some(preset) =
                                    find_builtin_preset_for_prompt(&st.system_prompt_template)
                                {
                                    st.system_prompt_template =
                                        preset.prompt_for_lang("en").to_string();
                                }
                            }
                        });
                });
                ui.add_space(6.0);
                ui.separator();
                ui.label(RichText::new(t!("system.prompt_section").as_ref()).strong());

                let builtins = builtin_prompt_presets();
                let lang_now = st.language.clone();
                let selected_label = builtins
                    .iter()
                    .find(|p| p.name == st.selected_prompt_preset)
                    .map(|p| p.display_name_for_lang(&lang_now).to_string())
                    .unwrap_or_else(|| st.selected_prompt_preset.clone());
                ui.horizontal(|ui| {
                    ui.label(t!("system.prompt_preset").as_ref());
                    egui::ComboBox::from_id_source("system_prompt_preset_combo")
                        .selected_text(&selected_label)
                        .width(210.0)
                        .show_ui(ui, |ui| {
                            ui.label(
                                RichText::new(t!("system.prompt_builtin_group").as_ref())
                                    .small()
                                    .color(Color32::GRAY),
                            );
                            for preset in builtins {
                                let is_sel = st.selected_prompt_preset == preset.name;
                                if ui
                                    .selectable_label(is_sel, preset.display_name_for_lang(&lang_now))
                                    .clicked()
                                {
                                    st.selected_prompt_preset = preset.name.to_string();
                                    st.system_prompt_template =
                                        preset.prompt_for_lang(&lang_now).to_string();
                                }
                            }
                            if !st.custom_prompts.is_empty() {
                                ui.separator();
                                ui.label(
                                    RichText::new(t!("system.prompt_custom_group").as_ref())
                                        .small()
                                        .color(Color32::GRAY),
                                );
                                for custom in st.custom_prompts.clone() {
                                    let is_sel = st.selected_prompt_preset == custom.name;
                                    if ui.selectable_label(is_sel, &custom.name).clicked() {
                                        st.selected_prompt_preset = custom.name.clone();
                                        st.system_prompt_template = custom.prompt.clone();
                                        self.custom_prompt_name = custom.name;
                                    }
                                }
                            }
                        });
                });

                ui.add(
                    egui::TextEdit::multiline(&mut st.system_prompt_template)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                ui.label(
                    RichText::new(t!("system.prompt_placeholders").as_ref())
                        .small()
                        .color(Color32::GRAY),
                );

                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.custom_prompt_name)
                            .desired_width(160.0)
                            .hint_text(t!("system.prompt_name_hint").as_ref()),
                    );
                    let name_clean = self.custom_prompt_name.trim().to_string();
                    if ui
                        .add_enabled(
                            !name_clean.is_empty(),
                            egui::Button::new(t!("system.prompt_save").as_ref()),
                        )
                        .clicked()
                    {
                        let tpl = st.system_prompt_template.clone();
                        if let Some(existing) = st
                            .custom_prompts
                            .iter_mut()
                            .find(|p| p.name.eq_ignore_ascii_case(&name_clean))
                        {
                            existing.prompt = tpl;
                        } else {
                            st.custom_prompts.push(CustomPromptPreset {
                                name: name_clean.clone(),
                                prompt: tpl,
                            });
                        }
                        st.selected_prompt_preset = name_clean;
                    }

                    let active_is_custom = st
                        .custom_prompts
                        .iter()
                        .any(|p| p.name == st.selected_prompt_preset);
                    if ui
                        .add_enabled(
                            active_is_custom,
                            egui::Button::new(t!("system.prompt_delete").as_ref()),
                        )
                        .clicked()
                    {
                        let target = st.selected_prompt_preset.clone();
                        st.custom_prompts.retain(|p| p.name != target);
                        if let Some(first) = builtins.first() {
                            st.selected_prompt_preset = first.name.to_string();
                            st.system_prompt_template =
                                first.prompt_for_lang(&lang_now).to_string();
                        }
                        self.custom_prompt_name.clear();
                    }
                });

                ui.add_space(6.0);
                ui.separator();
            }

            ui.label(RichText::new(t!("system.telemetry_section").as_ref()).strong());
            ui.add(
                egui::ProgressBar::new((snap.cpu_usage_percent / 100.0).clamp(0.0, 1.0))
                    .text(format!("CPU: {:.1}%", snap.cpu_usage_percent)),
            );
            let ram_ratio = if snap.total_memory_mb > 0 {
                snap.used_memory_mb as f32 / snap.total_memory_mb as f32
            } else {
                0.0
            };
            ui.add(egui::ProgressBar::new(ram_ratio.clamp(0.0, 1.0)).text(format!(
                "RAM: {} / {} MB (Proc: {} MB)",
                snap.used_memory_mb, snap.total_memory_mb, snap.process_memory_mb
            )));
            let max_temp_str = format!("{:.1}", snap.max_temp_celsius);
            ui.label(t!("system.max_temp", val = max_temp_str).as_ref());

            ui.add_space(6.0);
            let whisper_str = format!("{:.1}", whisper_acc);
            let ml_str = format!("{:.1}", ml_rate);
            ui.add(
                egui::ProgressBar::new((whisper_acc / 100.0).clamp(0.0, 1.0))
                    .text(t!("system.whisper_acc", val = whisper_str).as_ref()),
            );
            ui.add(
                egui::ProgressBar::new((ml_rate / 100.0).clamp(0.0, 1.0))
                    .text(t!("system.ml_rate", val = ml_str).as_ref()),
            );

            ui.add_space(6.0);
            {
                let mut st = self.shared.write();
                ui.checkbox(&mut st.power_save, t!("system.power_save").as_ref());
            }

            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new(t!("system.chart_title").as_ref()).strong());
            render_cpu_ram_chart(ui, &samples);

            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new(t!("system.latency_title").as_ref()).strong());
            if perf.is_empty() {
                ui.label(
                    RichText::new(t!("system.no_metrics").as_ref())
                        .small()
                        .color(Color32::GRAY),
                );
            } else {
                let max_avg = perf
                    .iter()
                    .map(|p| p.avg_ms)
                    .fold(1.0_f64, f64::max);
                for p in perf {
                    let frac = (p.avg_ms / max_avg).clamp(0.0, 1.0) as f32;
                    ui.add(egui::ProgressBar::new(frac).text(format!(
                        "{}: {:.1} ms (max {:.1} ms, {}x)",
                        p.operation, p.avg_ms, p.max_ms, p.count
                    )));
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new(t!("system.blind_vault").as_ref()).strong());
            ui.horizontal(|ui| {
                if ui.button(t!("system.export_diag").as_ref()).clicked() {
                    let _ = self.cmd_tx.send(CoreCommand::ExportDiag);
                }
                if ui.button(t!("system.run_dream").as_ref()).clicked() {
                    let _ = self.cmd_tx.send(CoreCommand::RunDreamReport);
                }
            });
            if let Some(report) = dream_report {
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(Color32::from_rgb(30, 34, 44))
                    .rounding(Rounding::same(6.0))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(RichText::new(report).small().color(Color32::LIGHT_GRAY))
                                .wrap(true),
                        );
                    });
            }
        });
    }

    fn render_tab_voz(&mut self, ui: &mut egui::Ui) {
        let mut tts_enabled = self.shared.read().tts_enabled;
        let model_path = self.shared.read().tts_model_path.clone();

        ui.label(RichText::new(t!("voice.title").as_ref()).strong());
        if ui
            .checkbox(&mut tts_enabled, t!("voice.enable_tts").as_ref())
            .changed()
        {
            self.shared.write().tts_enabled = tts_enabled;
            self.speech.set_enabled(tts_enabled);
        }

        ui.add_space(6.0);
        ui.label(
            RichText::new(t!("voice.current_onnx").as_ref())
                .small()
                .color(Color32::GRAY),
        );
        ui.label(RichText::new(&model_path).small().monospace());

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(t!("voice.select_onnx").as_ref()).clicked() {
                self.spawn_native_onnx_picker();
            }

            if ui.button(t!("voice.stop_speech").as_ref()).clicked() {
                self.speech.stop();
            }
        });
    }

    fn render_tab_logs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(t!("logs.search").as_ref());
            ui.add(
                egui::TextEdit::singleline(&mut self.log_search)
                    .desired_width(ui.available_width()),
            );
        });
        ui.separator();

        let query = self.log_search.to_lowercase();
        let events = EvolveMonitor::global().recent_events();
        let max_w = ui.available_width();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.set_max_width(max_w);
                for ev in events {
                    if !query.is_empty()
                        && !ev.message.to_lowercase().contains(&query)
                        && !ev.module.to_lowercase().contains(&query)
                    {
                        continue;
                    }

                    let ts = ev.timestamp.format("%H:%M:%S");
                    let level_col = match ev.level {
                        HealthLevel::Info => Color32::from_rgb(170, 210, 255),
                        HealthLevel::Warn => Color32::from_rgb(255, 200, 80),
                        HealthLevel::Error | HealthLevel::Fatal => Color32::from_rgb(255, 100, 100),
                    };

                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(format!("[{}]", ts)).small().color(Color32::GRAY));
                        ui.label(
                            RichText::new(format!("[{}]", ev.module))
                                .small()
                                .italics()
                                .color(Color32::from_rgb(110, 175, 255)),
                        );

                        // Destaque em itálico azul para tags internas como [HEALTH], [LEARN], [TTS]
                        if ev.message.starts_with('[') {
                            if let Some(end_idx) = ev.message.find(']') {
                                let tag = &ev.message[..=end_idx];
                                let rest = ev.message[end_idx + 1..].trim_start();
                                ui.label(
                                    RichText::new(tag)
                                        .small()
                                        .italics()
                                        .color(Color32::from_rgb(110, 175, 255)),
                                );
                                ui.add(
                                    egui::Label::new(RichText::new(rest).small().color(level_col))
                                        .wrap(true),
                                );
                            } else {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&ev.message).small().color(level_col),
                                    )
                                    .wrap(true),
                                );
                            }
                        } else {
                            ui.add(
                                egui::Label::new(
                                    RichText::new(&ev.message).small().color(level_col),
                                )
                                .wrap(true),
                            );
                        }
                    });
                }
            });
    }
}

impl eframe::App for ThrediaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        self.drain_stream();
        self.drain_file_picks();
        self.auto_save_settings_if_changed();

        let state_snapshot = self.shared.read().clone();

        egui::SidePanel::right("sidebar_panel")
            .resizable(false)
            .exact_width(320.0)
            .show(ctx, |ui| {
                ui.set_max_width(304.0);
                self.render_sidebar(ui);
            });

        egui::TopBottomPanel::top("header_panel")
            .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(egui::Margin::same(10.0)))
            .show(ctx, |ui| {
                self.render_header(ui, &state_snapshot);
            });

        egui::TopBottomPanel::bottom("input_panel")
            .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(egui::Margin::same(10.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let input_width = (ui.available_width() - 84.0).max(120.0);
                    let resp = ui.add_sized(
                        [input_width, 32.0],
                        egui::TextEdit::singleline(&mut self.input_buf)
                            .hint_text(t!("chat.input_hint").as_ref()),
                    );
                    let enter_pressed =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui
                        .add_sized([74.0, 32.0], egui::Button::new(t!("chat.send").as_ref()))
                        .clicked()
                        || enter_pressed
                    {
                        self.send_current_input();
                        resp.request_focus();
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_chat_area(ui, state_snapshot.show_thinking);
        });

        if self.smoke_test && self.frames >= 5 {
            EvolveMonitor::global().record_event(
                HealthLevel::Info,
                "core",
                format!(
                    "[SMOKE_OK] Janela GUI completa v{} aberta e renderizada ({} frames)",
                    env!("CARGO_PKG_VERSION"),
                    self.frames
                ),
            );
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }
}

fn render_bubble(ui: &mut egui::Ui, is_user: bool, sender: &str, text: &str) {
    let fill = if is_user {
        Color32::from_rgb(0, 112, 243)
    } else {
        Color32::from_rgb(38, 42, 54)
    };
    let align = if is_user {
        egui::Align::RIGHT
    } else {
        egui::Align::LEFT
    };

    ui.with_layout(egui::Layout::top_down(align), |ui| {
        egui::Frame::none()
            .fill(fill)
            .rounding(Rounding::same(12.0))
            .inner_margin(egui::Margin::symmetric(12.0, 8.0))
            .show(ui, |ui| {
                ui.set_max_width((ui.available_width() * 0.78).max(200.0));
                ui.label(
                    RichText::new(sender)
                        .small()
                        .strong()
                        .color(Color32::from_rgb(210, 225, 255)),
                );
                ui.label(RichText::new(text).color(Color32::WHITE));
            });
    });
}

fn apply_dark_messenger_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = Color32::from_rgb(20, 22, 28);
    visuals.panel_fill = Color32::from_rgb(24, 27, 34);
    visuals.extreme_bg_color = Color32::from_rgb(16, 18, 23);
    visuals.widgets.inactive.rounding = Rounding::same(8.0);
    visuals.widgets.hovered.rounding = Rounding::same(8.0);
    visuals.widgets.active.rounding = Rounding::same(8.0);
    ctx.set_visuals(visuals);
}

fn render_cpu_ram_chart(ui: &mut egui::Ui, samples: &[evolve::TelemetrySample]) {
    let desired = Vec2::new(ui.available_width().max(180.0), 84.0);
    let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
    let rect = response.rect;

    painter.rect_filled(rect, Rounding::same(6.0), Color32::from_rgb(18, 20, 26));
    painter.rect_stroke(
        rect,
        Rounding::same(6.0),
        Stroke::new(1.0_f32, Color32::from_rgb(48, 54, 68)),
    );

    if samples.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            t!("system.collecting_samples").as_ref(),
            egui::FontId::proportional(11.0),
            Color32::GRAY,
        );
        return;
    }

    let max_ram = samples
        .iter()
        .map(|s| s.process_ram_mb)
        .max()
        .unwrap_or(1)
        .max(128) as f32;

    let n = samples.len() as f32;
    let mut cpu_pts = Vec::with_capacity(samples.len());
    let mut ram_pts = Vec::with_capacity(samples.len());

    for (i, s) in samples.iter().enumerate() {
        let x = rect.left() + 6.0 + (i as f32 / (n - 1.0)) * (rect.width() - 12.0);
        let cpu_y = rect.bottom()
            - 6.0
            - (s.cpu_percent.clamp(0.0, 100.0) / 100.0) * (rect.height() - 12.0);
        let ram_y = rect.bottom()
            - 6.0
            - ((s.process_ram_mb as f32 / max_ram).clamp(0.0, 1.0)) * (rect.height() - 12.0);
        cpu_pts.push(egui::pos2(x, cpu_y));
        ram_pts.push(egui::pos2(x, ram_y));
    }

    painter.add(egui::Shape::line(
        ram_pts,
        Stroke::new(1.8_f32, Color32::from_rgb(100, 181, 246)),
    ));
    painter.add(egui::Shape::line(
        cpu_pts,
        Stroke::new(1.8_f32, Color32::from_rgb(76, 217, 100)),
    ));
}

fn open_memory_sql_file() {
    std::thread::spawn(|| {
        let db_path = memory::Memory::default_db_path();
        if !db_path.exists() {
            EvolveMonitor::global().record_event(
                HealthLevel::Warn,
                "memory",
                format!("Arquivo SQLite não encontrado em: {}", db_path.display()),
            );
            return;
        }

        #[cfg(target_os = "linux")]
        let res = std::process::Command::new("xdg-open").arg(&db_path).spawn();

        #[cfg(target_os = "windows")]
        let res = std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(&db_path)
            .spawn();

        #[cfg(target_os = "macos")]
        let res = std::process::Command::new("open").arg(&db_path).spawn();

        match res {
            Ok(_) => {
                EvolveMonitor::global().record_event(
                    HealthLevel::Info,
                    "memory",
                    format!("Abrindo banco SQLite: {}", db_path.display()),
                );
            }
            Err(e) => {
                EvolveMonitor::global().record_event(
                    HealthLevel::Warn,
                    "memory",
                    format!("Falha ao abrir SQLite ({}): {}", db_path.display(), e),
                );
            }
        }
    });
}
