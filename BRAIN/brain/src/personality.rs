use crate::state::EmotionState;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptFormat {
    ChatML,
    Llama3,
    TinyLlama,
    Gemma,
    Mistral,
}

impl PromptFormat {
    pub fn from_preset_name(preset: &str) -> Self {
        let lower = preset.to_lowercase();
        if lower.contains("llama 3") || lower.contains("llama3") {
            Self::Llama3
        } else if lower.contains("tinyllama") || lower.contains("zephyr") {
            Self::TinyLlama
        } else if lower.contains("gemma") {
            Self::Gemma
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            Self::Mistral
        } else {
            Self::ChatML
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelPromptPreset {
    pub name: &'static str,
    pub name_en: &'static str,
    pub format: PromptFormat,
    pub default_prompt: &'static str,
    pub default_prompt_en: &'static str,
}

impl ModelPromptPreset {
    pub fn prompt_for_lang(&self, lang: &str) -> &'static str {
        if lang.trim().to_ascii_lowercase().starts_with("en") {
            self.default_prompt_en
        } else {
            self.default_prompt
        }
    }

    pub fn display_name_for_lang(&self, lang: &str) -> &'static str {
        if lang.trim().to_ascii_lowercase().starts_with("en") {
            self.name_en
        } else {
            self.name
        }
    }
}

const BUILTIN_PRESETS: &[ModelPromptPreset] = &[
    ModelPromptPreset {
        name: "Padrão (Theia / ChatML)",
        name_en: "Default (Theia / ChatML)",
        format: PromptFormat::ChatML,
        default_prompt: "Você é {name}, {role}.\nResponda em português do Brasil de forma natural, direta e concisa (máximo 3 frases).\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}.\nReply in English in a natural, direct, and concise way (maximum 3 sentences).\nCurrent mood: {mood}.",
    },
    ModelPromptPreset {
        name: "ChatML (Qwen / SmolLM / Yi)",
        name_en: "ChatML (Qwen / SmolLM / Yi)",
        format: PromptFormat::ChatML,
        default_prompt: "Você é {name}, {role}. Seja claro, útil e objetivo.\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}. Be clear, helpful, and concise.\nCurrent mood: {mood}.",
    },
    ModelPromptPreset {
        name: "Llama 3 / 3.2 Instruct",
        name_en: "Llama 3 / 3.2 Instruct",
        format: PromptFormat::Llama3,
        default_prompt: "Você é {name}, {role}. Responda de forma natural, direta e concisa.\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}. Reply in a natural, direct, and concise way.\nCurrent mood: {mood}.",
    },
    ModelPromptPreset {
        name: "TinyLlama / Zephyr",
        name_en: "TinyLlama / Zephyr",
        format: PromptFormat::TinyLlama,
        default_prompt: "Você é {name}, {role}. Responda com empatia e clareza.\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}. Reply with empathy and clarity.\nCurrent mood: {mood}.",
    },
    ModelPromptPreset {
        name: "Gemma 2 / 3 Instruct",
        name_en: "Gemma 2 / 3 Instruct",
        format: PromptFormat::Gemma,
        default_prompt: "Você é {name}, {role}. Mantenha respostas diretas e úteis.\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}. Keep your answers direct and helpful.\nCurrent mood: {mood}.",
    },
    ModelPromptPreset {
        name: "Mistral / Mixtral Instruct",
        name_en: "Mistral / Mixtral Instruct",
        format: PromptFormat::Mistral,
        default_prompt: "Você é {name}, {role}. Seja preciso e conciso.\nHumor atual: {mood}.",
        default_prompt_en: "You are {name}, {role}. Be accurate and concise.\nCurrent mood: {mood}.",
    },
];

pub fn builtin_prompt_presets() -> &'static [ModelPromptPreset] {
    BUILTIN_PRESETS
}

pub fn find_builtin_preset_for_prompt(prompt: &str) -> Option<&'static ModelPromptPreset> {
    let trimmed = prompt.trim();
    BUILTIN_PRESETS
        .iter()
        .find(|p| p.default_prompt.trim() == trimmed || p.default_prompt_en.trim() == trimmed)
}

/// Compositor de Prompt (ChatML e templates de outros modelos) injetando identidade,
/// regras concisas, humor VAD, contexto RAG (`retrieval_context`) e histórico recente.
pub struct PromptComposer;

impl PromptComposer {
    pub fn compose_chatml(
        identity: &HashMap<String, String>,
        emotion: &EmotionState,
        retrieval_context: &[String],
        recent_history_chatml: &str,
        user_input: &str,
    ) -> String {
        Self::compose_chatml_with_lang(
            identity,
            emotion,
            retrieval_context,
            recent_history_chatml,
            user_input,
            "pt-BR",
        )
    }

    pub fn compose_chatml_with_lang(
        identity: &HashMap<String, String>,
        emotion: &EmotionState,
        retrieval_context: &[String],
        recent_history_chatml: &str,
        user_input: &str,
        lang: &str,
    ) -> String {
        Self::compose_with_template(
            identity,
            emotion,
            retrieval_context,
            recent_history_chatml,
            user_input,
            lang,
            None,
            "Padrão (Theia / ChatML)",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compose_with_template(
        identity: &HashMap<String, String>,
        emotion: &EmotionState,
        retrieval_context: &[String],
        recent_history_chatml: &str,
        user_input: &str,
        lang: &str,
        custom_template: Option<&str>,
        preset_name: &str,
    ) -> String {
        let is_en = lang.trim().to_ascii_lowercase().starts_with("en");
        let name = identity
            .get("nome")
            .or_else(|| identity.get("name"))
            .map(String::as_str)
            .unwrap_or("Theia");
        let default_role = if is_en {
            "Local cognitive assistant in Rust"
        } else {
            "Assistente cognitiva local em Rust"
        };
        let role = identity
            .get("papel")
            .or_else(|| identity.get("role"))
            .map(String::as_str)
            .unwrap_or(default_role);

        let mood = emotion.get_mood_desc_for(lang);
        let mut system_block = if let Some(tpl) = custom_template.map(str::trim).filter(|s| !s.is_empty()) {
            let effective_tpl = if let Some(preset) = find_builtin_preset_for_prompt(tpl) {
                preset.prompt_for_lang(lang)
            } else {
                tpl
            };
            let mut rendered = effective_tpl
                .replace("{name}", name)
                .replace("{role}", role)
                .replace("{mood}", &mood);
            if !rendered.ends_with('\n') {
                rendered.push('\n');
            }
            rendered
        } else if is_en {
            format!(
                "You are {name}, {role}.\n\
                 Reply in English in a natural, direct, and concise way (maximum 3 sentences).\n\
                 Current mood: {mood}.\n",
                name = name,
                role = role,
                mood = mood,
            )
        } else {
            format!(
                "Você é {name}, {role}.\n\
                 Responda em português do Brasil de forma natural, direta e concisa (máximo 3 frases).\n\
                 Humor atual: {mood}.\n",
                name = name,
                role = role,
                mood = mood,
            )
        };

        if !retrieval_context.is_empty() {
            system_block.push_str(if is_en {
                "Relevant facts:\n"
            } else {
                "Fatos relevantes:\n"
            });
            for item in retrieval_context {
                system_block.push_str(&format!("- {}\n", item));
            }
        }

        let format = PromptFormat::from_preset_name(preset_name);
        format_prompt_by_model(format, &system_block, recent_history_chatml, user_input.trim())
    }
}

fn format_prompt_by_model(
    format: PromptFormat,
    system_block: &str,
    recent_history_chatml: &str,
    user_input: &str,
) -> String {
    match format {
        PromptFormat::ChatML => {
            let mut prompt = String::with_capacity(
                system_block.len() + recent_history_chatml.len() + user_input.len() + 128,
            );
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system_block);
            prompt.push_str("<|im_end|>\n");
            if !recent_history_chatml.is_empty() {
                prompt.push_str(recent_history_chatml);
                if !recent_history_chatml.ends_with('\n') {
                    prompt.push('\n');
                }
            }
            prompt.push_str("<|im_start|>user\n");
            prompt.push_str(user_input);
            prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
            prompt
        }
        PromptFormat::Llama3 => {
            let history = convert_chatml_history(recent_history_chatml, |role, content| {
                format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role, content
                )
            });
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                system_block.trim_end(),
                history,
                user_input
            )
        }
        PromptFormat::TinyLlama => {
            let history = convert_chatml_history(recent_history_chatml, |role, content| {
                format!("<|{}|>\n{}</s>\n", role, content)
            });
            format!(
                "<|system|>\n{}</s>\n{}<|user|>\n{}</s>\n<|assistant|>\n",
                system_block.trim_end(),
                history,
                user_input
            )
        }
        PromptFormat::Gemma => {
            let history = convert_chatml_history(recent_history_chatml, |role, content| {
                let gemma_role = if role == "assistant" { "model" } else { "user" };
                format!("<start_of_turn>{}\n{}<end_of_turn>\n", gemma_role, content)
            });
            format!(
                "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
                system_block.trim_end(),
                if history.is_empty() {
                    user_input.to_string()
                } else {
                    format!("{}{}", history, user_input)
                }
            )
        }
        PromptFormat::Mistral => {
            let history = convert_chatml_history(recent_history_chatml, |role, content| {
                if role == "user" {
                    format!("[INST] {} [/INST] ", content)
                } else {
                    format!("{}</s> ", content)
                }
            });
            format!(
                "[INST] {}\n\n{} [/INST]",
                system_block.trim_end(),
                if history.is_empty() {
                    user_input.to_string()
                } else {
                    format!("{}{}", history, user_input)
                }
            )
        }
    }
}

fn convert_chatml_history<F>(history_chatml: &str, mut fmt_turn: F) -> String
where
    F: FnMut(&str, &str) -> String,
{
    let mut out = String::new();
    for block in history_chatml.split("<|im_start|>") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((role, rest)) = trimmed.split_once('\n') {
            let content = rest.split("<|im_end|>").next().unwrap_or(rest).trim();
            if !content.is_empty() {
                out.push_str(&fmt_turn(role.trim(), content));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_prompt_and_model_formats() {
        let mut id = HashMap::new();
        id.insert("nome".to_string(), "Theia".to_string());
        let emo = EmotionState::default();
        let hist = "<|im_start|>user\nOi<|im_end|>\n<|im_start|>assistant\nOlá!<|im_end|>\n";

        let llama3 = PromptComposer::compose_with_template(
            &id,
            &emo,
            &[],
            hist,
            "Tudo bem?",
            "pt-BR",
            Some("Sou {name}."),
            "Llama 3 / 3.2 Instruct",
        );
        assert!(llama3.contains("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"));
        assert!(llama3.contains("Sou Theia."));
        assert!(llama3.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        let gemma = PromptComposer::compose_with_template(
            &id,
            &emo,
            &[],
            "",
            "Oi",
            "pt-BR",
            Some("Prompt Gemma"),
            "Gemma 2 / 3 Instruct",
        );
        assert!(gemma.contains("<start_of_turn>user\nPrompt Gemma"));
        assert!(gemma.ends_with("<start_of_turn>model\n"));
    }
}
