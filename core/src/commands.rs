use brain::ComputeBackend;
use memory::{ChatMessage, Fact};
use parking_lot::RwLock;
use regex::Regex;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    pub facts: Vec<Fact>,
    pub messages: Vec<ChatMessage>,
}

pub type MemorySnapshotHandle = Arc<RwLock<MemorySnapshot>>;

#[derive(Debug, Clone)]
pub enum CoreCommand {
    UserMessage(String),
    HotSwapLlm {
        model_path: String,
        n_ctx: u32,
        backend: ComputeBackend,
        move_to_modelos: bool,
    },
    HotSwapSlm {
        model_path: String,
        move_to_modelos: bool,
    },
    RefreshModels,
    RefreshMemoryView,
    AddFact {
        subject: String,
        attribute: String,
        value: String,
        confidence: f64,
    },
    UpdateFact {
        id: i64,
        subject: String,
        attribute: String,
        value: String,
        confidence: f64,
    },
    DeleteFact {
        id: i64,
    },
    DeleteChatMessage {
        id: i64,
    },
    ClearChatHistory,
    ExportDiag,
    RunDreamReport,
    SaveSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemIntent {
    Status,
    DebugMemory,
    ClearHistory,
    Help,
}

/// Dispatcher Zero-Latency baseado em Regex compiladas uma única vez.
pub struct IntentDispatcher;

impl IntentDispatcher {
    pub fn dispatch(input: &str) -> Option<SystemIntent> {
        static STATUS_RE: OnceLock<Regex> = OnceLock::new();
        static DEBUG_MEM_RE: OnceLock<Regex> = OnceLock::new();
        static CLEAR_RE: OnceLock<Regex> = OnceLock::new();
        static HELP_RE: OnceLock<Regex> = OnceLock::new();

        let status_re = STATUS_RE
            .get_or_init(|| Regex::new(r"(?i)^\s*/?(status|sistema|system|hardware)\s*$").unwrap());
        let debug_mem_re = DEBUG_MEM_RE
            .get_or_init(|| Regex::new(r"(?i)^\s*/?(debug\s+mem|listar\s+fatos|list\s+facts)\s*$").unwrap());
        let clear_re = CLEAR_RE
            .get_or_init(|| Regex::new(r"(?i)^\s*/?(limpar|esquecer|resetar|clear|forget|reset)\s*$").unwrap());
        let help_re = HELP_RE
            .get_or_init(|| Regex::new(r"(?i)^\s*/?(ajuda|help)\s*$").unwrap());

        if status_re.is_match(input) {
            Some(SystemIntent::Status)
        } else if debug_mem_re.is_match(input) {
            Some(SystemIntent::DebugMemory)
        } else if clear_re.is_match(input) {
            Some(SystemIntent::ClearHistory)
        } else if help_re.is_match(input) {
            Some(SystemIntent::Help)
        } else {
            None
        }
    }

    pub fn help_text_for(lang: &str) -> &'static str {
        if lang.trim().to_ascii_lowercase().starts_with("en") {
            "Quick commands (Zero-Latency):\n\
             • status / system / hardware — displays CPU, RAM, temperature and VAD telemetry\n\
             • debug mem / list facts — lists all facts stored in SQLite\n\
             • clear / forget / reset — clears SQLite chat history and KV Cache\n\
             • help — shows this command list"
        } else {
            "Comandos rápidos (Zero-Latency):\n\
             • status / sistema / hardware — exibe telemetria de CPU, RAM, temperatura e VAD\n\
             • debug mem / listar fatos — lista todos os fatos armazenados no SQLite\n\
             • limpar / esquecer / resetar — limpa o histórico de chat no SQLite e o KV Cache\n\
             • ajuda / help — mostra esta lista de comandos"
        }
    }
}

/// Remove blocos `<think>...</think>` (inclusive incompletos), tags `<|...|>` / `<|im_end|`,
/// sangramentos de template (`<|eot_id|>`, `<end_of_turn>`, `</s>`, `[/INST]`) e `[LEARN:...]`
/// preservando texto e números normais.
pub fn sanitize_assistant_text(raw: &str, keep_thinking: bool) -> String {
    static THINK_COMPLETE_RE: OnceLock<Regex> = OnceLock::new();
    static THINK_OPEN_RE: OnceLock<Regex> = OnceLock::new();
    static TURN_BLEED_RE: OnceLock<Regex> = OnceLock::new();
    static SPECIAL_TAG_RE: OnceLock<Regex> = OnceLock::new();
    static SPECIAL_INCOMPLETE_RE: OnceLock<Regex> = OnceLock::new();
    static TEMPLATE_TOKENS_RE: OnceLock<Regex> = OnceLock::new();
    static LEARN_TAG_RE: OnceLock<Regex> = OnceLock::new();
    static STRAY_CLOSE_RE: OnceLock<Regex> = OnceLock::new();

    let think_complete = THINK_COMPLETE_RE.get_or_init(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    let think_open = THINK_OPEN_RE.get_or_init(|| Regex::new(r"(?s)<think>.*$").unwrap());
    let turn_bleed = TURN_BLEED_RE.get_or_init(|| {
        Regex::new(r"(?is)(?:<\s*\|\s*im_start\|?|<\s*\|\s*im_end\|?|<\s*\|\s*eot_id\|?|<\s*\|\s*end_of_text\|?|<\s*\|\s*endoftext\|?|<\s*\|\s*start_header_id\|?|<\s*\|\s*end_header_id\|?|<end_of_turn>|<start_of_turn>).*$").unwrap()
    });
    let special_tag = SPECIAL_TAG_RE.get_or_init(|| Regex::new(r"<\s*\|[^|\n]*\|\s*>").unwrap());
    let special_incomplete = SPECIAL_INCOMPLETE_RE.get_or_init(|| Regex::new(r"<\s*\|[^\n>]*$").unwrap());
    let template_tokens = TEMPLATE_TOKENS_RE
        .get_or_init(|| Regex::new(r"(?i)(?:</?s>|\[/?INST\])").unwrap());
    let learn_tag = LEARN_TAG_RE.get_or_init(|| Regex::new(r"(?i)\[LEARN:[^\]]*\]").unwrap());
    let stray_close = STRAY_CLOSE_RE
        .get_or_init(|| Regex::new(r"(?:</?|<\s*\|?|\|\s*>?|\s+>|\n>)\s*$").unwrap());

    let mut text = raw.to_string();
    if !keep_thinking {
        text = think_complete.replace_all(&text, "").into_owned();
        text = think_open.replace_all(&text, "").into_owned();
    }
    text = turn_bleed.replace_all(&text, "").into_owned();
    text = special_tag.replace_all(&text, "").into_owned();
    text = special_incomplete.replace_all(&text, "").into_owned();
    text = template_tokens.replace_all(&text, "").into_owned();
    text = learn_tag.replace_all(&text, "").into_owned();
    text = stray_close.replace_all(&text, "").into_owned();
    let trimmed = text
        .trim_end_matches(|c: char| c.is_whitespace() || c == '<' || c == '>' || c == '|')
        .trim_end_matches("</")
        .trim_end_matches(|c: char| c.is_whitespace() || c == '<' || c == '>' || c == '|');
    trimmed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_dispatcher() {
        assert_eq!(IntentDispatcher::dispatch("status"), Some(SystemIntent::Status));
        assert_eq!(IntentDispatcher::dispatch("/hardware"), Some(SystemIntent::Status));
        assert_eq!(IntentDispatcher::dispatch("debug mem"), Some(SystemIntent::DebugMemory));
        assert_eq!(IntentDispatcher::dispatch("listar fatos"), Some(SystemIntent::DebugMemory));
        assert_eq!(IntentDispatcher::dispatch("limpar"), Some(SystemIntent::ClearHistory));
        assert_eq!(IntentDispatcher::dispatch("help"), Some(SystemIntent::Help));
        assert_eq!(IntentDispatcher::dispatch("Olá, tudo bem?"), None);
    }

    #[test]
    fn test_sanitize_preserves_numbers_and_strips_tags() {
        let raw = "<think>raciocinio 123</think>Olá! Tenho 42 anos e 3.14 pontos. [LEARN: User | idade | 42]<|im_end|>";
        let clean = sanitize_assistant_text(raw, false);
        assert_eq!(clean, "Olá! Tenho 42 anos e 3.14 pontos.");

        let incomplete = "<think>ainda pensando...";
        assert_eq!(sanitize_assistant_text(incomplete, false), "");

        let kept = sanitize_assistant_text(raw, true);
        assert!(kept.contains("<think>raciocinio 123</think>"));
        assert!(!kept.contains("[LEARN:"));

        assert_eq!(sanitize_assistant_text("Oi!<|im_end|", false), "Oi!");
        assert_eq!(
            sanitize_assistant_text("Tudo bem<|im_start|>user\noi", false),
            "Tudo bem"
        );
        assert_eq!(sanitize_assistant_text("Ok<|eot_id", false), "Ok");
        assert_eq!(sanitize_assistant_text("Olá!<", false), "Olá!");
        assert_eq!(sanitize_assistant_text("Oi! <", false), "Oi!");
        assert_eq!(sanitize_assistant_text("Fim</", false), "Fim");
        assert_eq!(
            sanitize_assistant_text("Em programação? < |im_end", false),
            "Em programação?"
        );
        assert_eq!(
            sanitize_assistant_text("Precisando de ajuda? |", false),
            "Precisando de ajuda?"
        );
        assert_eq!(sanitize_assistant_text("Olá!>", false), "Olá!");
        assert_eq!(sanitize_assistant_text("Olá! > >", false), "Olá!");
        assert_eq!(sanitize_assistant_text("2 > 1.", false), "2 > 1.");
    }
}

