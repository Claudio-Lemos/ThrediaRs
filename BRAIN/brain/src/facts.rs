use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Pontuação mínima de relevância para persistir um fato na memória de longo prazo.
pub const MIN_FACT_RELEVANCE: f64 = 0.60;

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    pub subject: String,
    pub attribute: String,
    pub value: String,
    pub relevance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelevanceAssessment {
    pub score: f64,
    pub is_relevant: bool,
    pub reason: String,
}

/// Sistema de Relevância: avalia se a entrada do usuário contém informação
/// persistente relevante, pontua candidatos extraídos e descarta ruído/chitchat.
pub struct RelevanceSystem;

impl RelevanceSystem {
    /// Avalia se a mensagem do usuário possui conteúdo declarativo potencialmente relevante.
    pub fn assess_user_input(user_input: &str) -> RelevanceAssessment {
        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            return RelevanceAssessment {
                score: 0.0,
                is_relevant: false,
                reason: "entrada vazia".to_string(),
            };
        }

        if has_learn_tag(trimmed) {
            return RelevanceAssessment {
                score: 0.95,
                is_relevant: true,
                reason: "tag explícita [LEARN]".to_string(),
            };
        }

        let declarative_clauses = split_declarative_clauses(trimmed);
        if declarative_clauses.is_empty() {
            return RelevanceAssessment {
                score: 0.10,
                is_relevant: false,
                reason: "apenas perguntas ou ruído fático".to_string(),
            };
        }

        let mut best_score: f64 = 0.0;
        let mut best_reason = "sem marcadores factuais".to_string();

        for clause in &declarative_clauses {
            let lower = clause.to_lowercase();
            if is_phatic_or_command(&lower) {
                continue;
            }

            let word_count = lower
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 2)
                .count();

            if word_count < 2 {
                continue;
            }

            let has_personal_marker = [
                "meu ",
                "minha ",
                "me chamo",
                "sou ",
                "eu ",
                "moro ",
                "trabalho ",
                "estudo ",
                "gosto ",
                "adoro ",
                "amo ",
                "curto ",
                "prefiro ",
                "odeio ",
                "detesto ",
                "não gosto",
                "nao gosto",
                "tenho ",
                "uso ",
                "falo ",
                "quero ",
                "pretendo ",
                "seu nome",
                "te chamo",
                "my ",
                "i am ",
                "i'm ",
                "i live",
                "i work",
                "i study",
                "i like",
                "i love",
                "i enjoy",
                "i prefer",
                "i hate",
                "i have ",
                "i use ",
                "i speak ",
                "your name",
            ]
            .iter()
            .any(|m| lower.contains(m));

            let score = if has_personal_marker {
                0.85
            } else if word_count >= 5 {
                0.55
            } else {
                0.35
            };

            if score > best_score {
                best_score = score;
                best_reason = if has_personal_marker {
                    "cláusula declarativa com marcador pessoal/contextual".to_string()
                } else {
                    "cláusula informativa".to_string()
                };
            }
        }

        RelevanceAssessment {
            score: best_score,
            is_relevant: best_score >= 0.45,
            reason: best_reason,
        }
    }

    /// Filtra ruídos, pontua relevância, deduplica e ordena fatos extraídos.
    pub fn score_and_filter(candidates: Vec<ExtractedFact>) -> Vec<ExtractedFact> {
        let mut dedup: HashMap<(String, String, String), ExtractedFact> = HashMap::new();

        for mut fact in candidates {
            fact.subject = normalize_field(&fact.subject);
            fact.attribute = normalize_field(&fact.attribute);
            fact.value = clean_value(&fact.value);

            if !is_valid_fact(&fact) {
                continue;
            }

            let base_score = category_base_relevance(&fact.attribute);
            let combined = (fact.relevance * 0.4 + base_score * 0.6).clamp(0.0, 1.0);
            fact.relevance = combined;

            if fact.relevance < MIN_FACT_RELEVANCE {
                continue;
            }

            let key = (
                fact.subject.to_lowercase(),
                fact.attribute.to_lowercase(),
                fact.value.to_lowercase(),
            );
            dedup
                .entry(key)
                .and_modify(|existing| {
                    if fact.relevance > existing.relevance {
                        existing.relevance = fact.relevance;
                    }
                })
                .or_insert(fact);
        }

        let mut result: Vec<ExtractedFact> = dedup.into_values().collect();
        result.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.attribute.cmp(&b.attribute))
                .then_with(|| a.value.cmp(&b.value))
        });
        result
    }
}

/// Extrai tags explícitas `[LEARN: Sujeito | atributo | valor]` de um texto.
pub fn parse_learn_tags(text: &str, base_relevance: f64) -> Vec<ExtractedFact> {
    static LEARN_RE: OnceLock<Regex> = OnceLock::new();
    let learn_re = LEARN_RE.get_or_init(|| {
        Regex::new(r"(?i)\[LEARN:\s*([^|\]]+?)\s*\|\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]").unwrap()
    });

    learn_re
        .captures_iter(text)
        .map(|cap| ExtractedFact {
            subject: cap[1].trim().to_string(),
            attribute: cap[2].trim().to_string(),
            value: cap[3].trim().to_string(),
            relevance: base_relevance,
        })
        .collect()
}

/// Extrai fatos no formato simples `atributo: valor` gerados por SLMs leves.
/// Linhas `NONE` / `NADA` interrompem ou são ignoradas.
pub fn parse_slm_kv_lines(text: &str, base_relevance: f64) -> Vec<ExtractedFact> {
    let mut facts = parse_learn_tags(text, base_relevance);
    for raw_line in text.lines() {
        let line = raw_line
            .trim()
            .trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_whitespace());
        if line.is_empty()
            || line.eq_ignore_ascii_case("none")
            || line.eq_ignore_ascii_case("nada")
            || line.starts_with('[')
        {
            continue;
        }
        // Para de ler se o modelo começar a gerar outro exemplo ("Text:" / "Entrada:")
        let lower_line = line.to_lowercase();
        if lower_line.starts_with("text:")
            || lower_line.starts_with("entrada:")
            || lower_line.starts_with("facts:")
        {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let attr = k.trim().to_lowercase().replace(' ', "_");
            let val = v.trim().trim_matches('"').trim_matches('\'').trim();
            let is_prompt_echo_key = matches!(
                attr.as_str(),
                "mensagem"
                    | "message"
                    | "resposta"
                    | "reply"
                    | "exemplo"
                    | "example"
                    | "texto"
                    | "text"
                    | "entrada"
                    | "saida"
                    | "saída"
                    | "input"
                    | "output"
                    | "fato"
                    | "facts"
            );
            if !is_prompt_echo_key
                && !attr.is_empty()
                && !val.is_empty()
                && !val.contains('?')
                && !val.eq_ignore_ascii_case("none")
                && !val.eq_ignore_ascii_case("nada")
            {
                facts.push(ExtractedFact {
                    subject: "Usuario".to_string(),
                    attribute: attr,
                    value: val.to_string(),
                    relevance: base_relevance,
                });
            }
        }
    }
    facts
}

/// Extrai fatos declarativos relevantes da mensagem do usuário (PT-BR e EN).
pub fn extract_declarative_facts(user_input: &str) -> Vec<ExtractedFact> {
    let mut facts = parse_learn_tags(user_input, 0.95);
    let clauses = split_declarative_clauses(user_input);

    for clause in clauses {
        extract_from_clause(&clause, &mut facts);
    }

    facts
}

fn extract_from_clause(clause: &str, out: &mut Vec<ExtractedFact>) {
    static USER_NAME_RE: OnceLock<Regex> = OnceLock::new();
    static USER_SOU_NAME_RE: OnceLock<Regex> = OnceLock::new();
    static ASSISTANT_NAME_RE: OnceLock<Regex> = OnceLock::new();
    static LIVE_RE: OnceLock<Regex> = OnceLock::new();
    static ORIGIN_RE: OnceLock<Regex> = OnceLock::new();
    static WORK_RE: OnceLock<Regex> = OnceLock::new();
    static STUDY_RE: OnceLock<Regex> = OnceLock::new();
    static FAVORITE_RE: OnceLock<Regex> = OnceLock::new();
    static PREFER_RE: OnceLock<Regex> = OnceLock::new();
    static LIKE_RE: OnceLock<Regex> = OnceLock::new();
    static DISLIKE_RE: OnceLock<Regex> = OnceLock::new();
    static PROJECT_RE: OnceLock<Regex> = OnceLock::new();
    static GOAL_RE: OnceLock<Regex> = OnceLock::new();
    static AGE_RE: OnceLock<Regex> = OnceLock::new();
    static LANG_RE: OnceLock<Regex> = OnceLock::new();
    static PET_RE: OnceLock<Regex> = OnceLock::new();
    static TOOL_RE: OnceLock<Regex> = OnceLock::new();

    let user_name_re = USER_NAME_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:meu nome é|meu nome e|me chamo|pode me chamar de|my name is|i am called|call me)\s+([A-Za-zÀ-ÿ0-9_-]{2,32})").unwrap()
    });
    let user_sou_name_re = USER_SOU_NAME_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu sou [oa]|sou [oa])\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ_-]{1,30})\b").unwrap()
    });
    let assistant_name_re = ASSISTANT_NAME_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:seu nome (?:agora )?(?:é|e|será)|te chamo de|vou te chamar de|mude seu nome para|your name is|i'?ll call you)\s+([A-Za-zÀ-ÿ0-9_-]{2,32})").unwrap()
    });
    let live_re = LIVE_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu moro em|moro em|moro n[oa]|eu moro n[oa]|estou morando em|vivo em|i live in)\s+([A-Za-zÀ-ÿ0-9\s,'_-]{2,48})").unwrap()
    });
    let origin_re = ORIGIN_RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:sou de|nasci em|vim de|i'?m from|i am from)\s+([A-Za-zÀ-ÿ0-9\s,'_-]{2,48})",
        )
        .unwrap()
    });
    let work_re = WORK_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu trabalho como|trabalho como|trabalho com|minha profiss[aã]o [eé]|i work as)\s+([A-Za-zÀ-ÿ0-9\s+/#_-]{2,48})").unwrap()
    });
    let study_re = STUDY_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu estudo|estudo|faço faculdade de|faco faculdade de|sou estudante de|i study)\s+([A-Za-zÀ-ÿ0-9\s+/#_-]{2,48})").unwrap()
    });
    let favorite_re = FAVORITE_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:meu|minha)\s+([A-Za-zÀ-ÿ0-9_-]{2,24})\s+favorit[oa]\s+[eé]\s+([A-Za-zÀ-ÿ0-9\s+/#_-]{2,48})").unwrap()
    });
    let prefer_re = PREFER_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu prefiro|prefiro|i prefer)\s+([A-Za-zÀ-ÿ0-9\s+/#_-]{2,48})")
            .unwrap()
    });
    let like_re = LIKE_RE.get_or_init(|| {
        Regex::new(r"(?i)(?:\b(?:eu gosto de|gosto muito de|gosto de|adoro|amo|curto|i like|i love|i enjoy))\s+([A-Za-zÀ-ÿ0-9\s,+/#_-]{2,60})").unwrap()
    });
    let dislike_re = DISLIKE_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu n[aã]o gosto de|n[aã]o gosto de|nao gosto de|detesto|odeio|i don'?t like|i hate|i dislike)\s+([A-Za-zÀ-ÿ0-9\s,+/#_-]{2,48})").unwrap()
    });
    let project_re = PROJECT_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:meu projeto [eé]|estou criando|estou desenvolvendo|estou trabalhando (?:em|no|na)|i am building|i'?m working on)\s+([A-Za-zÀ-ÿ0-9\s+/#_.-]{2,60})").unwrap()
    });
    let goal_re = GOAL_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:meu objetivo [eé]|quero aprender|pretendo|i want to learn|my goal is)\s+([A-Za-zÀ-ÿ0-9\s+/#_-]{2,60})").unwrap()
    });
    let age_re = AGE_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu tenho|tenho|i am|i'?m)\s+(\d{1,3})\s+(?:anos|years old)\b")
            .unwrap()
    });
    let lang_re = LANG_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:eu falo|falo|i speak)\s+([A-Za-zÀ-ÿ0-9\s,_-]{2,40})").unwrap()
    });
    let pet_re = PET_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:tenho um|tenho uma|meu|minha)\s+(cachorro|c[aã]o|gato|gata|p[aá]ssaro|papagaio|dog|cat)\s+(?:chamado|chamada|se chama|named)\s+([A-Za-zÀ-ÿ0-9_-]{2,32})").unwrap()
    });
    let tool_re = TOOL_RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:meu sistema [eé]|eu uso|uso)\s+(Linux|Windows|macOS|Arch Linux|Ubuntu|Debian|Fedora|NixOS|Neovim|VS Code|Rust|Python)\b").unwrap()
    });

    if let Some(cap) = user_name_re.captures(clause) {
        push_fact(out, "Usuario", "nome", &cap[1], 0.98);
    } else if let Some(cap) = user_sou_name_re.captures(clause) {
        push_fact(out, "Usuario", "nome", &cap[1], 0.90);
    }

    if let Some(cap) = assistant_name_re.captures(clause) {
        push_fact(out, "Assistente", "nome", &cap[1], 0.98);
    }

    if let Some(cap) = live_re.captures(clause) {
        push_fact(
            out,
            "Usuario",
            "localizacao",
            &clean_clause_tail(&cap[1]),
            0.92,
        );
    }
    if let Some(cap) = origin_re.captures(clause) {
        push_fact(out, "Usuario", "origem", &clean_clause_tail(&cap[1]), 0.90);
    }
    if let Some(cap) = work_re.captures(clause) {
        push_fact(
            out,
            "Usuario",
            "profissao",
            &clean_clause_tail(&cap[1]),
            0.92,
        );
    }
    if let Some(cap) = study_re.captures(clause) {
        push_fact(out, "Usuario", "estudo", &clean_clause_tail(&cap[1]), 0.88);
    }
    if let Some(cap) = favorite_re.captures(clause) {
        let attr = format!("{}_favorito", cap[1].trim().to_lowercase());
        push_fact(out, "Usuario", &attr, &clean_clause_tail(&cap[2]), 0.90);
    }
    if let Some(cap) = prefer_re.captures(clause) {
        push_fact(
            out,
            "Usuario",
            "preferencia",
            &clean_clause_tail(&cap[1]),
            0.85,
        );
    }

    // Evita que "não gosto de X" dispare também "gosto de X"
    let has_neg_like = dislike_re.is_match(clause);
    if has_neg_like {
        if let Some(cap) = dislike_re.captures(clause) {
            push_fact(
                out,
                "Usuario",
                "desinteresse",
                &clean_clause_tail(&cap[1]),
                0.85,
            );
        }
    } else if let Some(cap) = like_re.captures(clause) {
        let raw_val = clean_clause_tail(&cap[1]);
        if raw_val.contains(',') {
            for item in raw_val.split(',') {
                let cleaned = clean_clause_tail(item);
                if !cleaned.is_empty() {
                    push_fact(out, "Usuario", "interesse", &cleaned, 0.85);
                }
            }
        } else {
            push_fact(out, "Usuario", "interesse", &raw_val, 0.85);
        }
    }

    if let Some(cap) = project_re.captures(clause) {
        push_fact(out, "Usuario", "projeto", &clean_clause_tail(&cap[1]), 0.90);
    }
    if let Some(cap) = goal_re.captures(clause) {
        push_fact(
            out,
            "Usuario",
            "objetivo",
            &clean_clause_tail(&cap[1]),
            0.86,
        );
    }
    if let Some(cap) = age_re.captures(clause) {
        if let Ok(age) = cap[1].parse::<u16>() {
            if (1..=120).contains(&age) {
                push_fact(out, "Usuario", "idade", &format!("{} anos", age), 0.92);
            }
        }
    }
    if let Some(cap) = lang_re.captures(clause) {
        push_fact(out, "Usuario", "idioma", &clean_clause_tail(&cap[1]), 0.88);
    }
    if let Some(cap) = pet_re.captures(clause) {
        let pet_val = format!("{} ({})", cap[1].trim().to_lowercase(), cap[2].trim());
        push_fact(out, "Usuario", "pet", &pet_val, 0.88);
    }
    if let Some(cap) = tool_re.captures(clause) {
        push_fact(out, "Usuario", "ferramenta", cap[1].trim(), 0.85);
    }
}

fn push_fact(
    out: &mut Vec<ExtractedFact>,
    subject: &str,
    attribute: &str,
    value: &str,
    relevance: f64,
) {
    out.push(ExtractedFact {
        subject: subject.to_string(),
        attribute: attribute.to_string(),
        value: value.to_string(),
        relevance,
    });
}

fn has_learn_tag(text: &str) -> bool {
    text.contains("[LEARN:") || text.contains("[learn:")
}

/// Divide a entrada em cláusulas declarativas, ignorando perguntas (`?`).
fn split_declarative_clauses(input: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    let mut current = String::new();

    for ch in input.chars() {
        match ch {
            '.' | '!' | ';' | '\n' => {
                let trimmed = current.trim();
                if !trimmed.is_empty() && !is_interrogative_clause(trimmed) {
                    clauses.push(trimmed.to_string());
                }
                current.clear();
            }
            '?' => {
                // Cláusula terminada em '?' é interrogativa e não deve gerar fatos declarativos.
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() && !is_interrogative_clause(trimmed) {
        clauses.push(trimmed.to_string());
    }

    clauses
}

fn is_interrogative_clause(clause: &str) -> bool {
    let lower = clause.trim().to_lowercase();
    [
        "qual ", "quais ", "quem ", "onde ", "quando ", "como ", "por que ", "porque ", "o que ",
        "what ", "who ", "where ", "when ", "how ", "why ",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn is_phatic_or_command(lower: &str) -> bool {
    let clean = lower.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ');
    matches!(
        clean.trim(),
        "oi" | "olá"
            | "ola"
            | "tudo bem"
            | "bom dia"
            | "boa tarde"
            | "boa noite"
            | "ok"
            | "blz"
            | "beleza"
            | "valeu"
            | "obrigado"
            | "obrigada"
            | "tchau"
            | "haha"
            | "kkkk"
            | "sim"
            | "não"
            | "nao"
            | "hi"
            | "hello"
            | "hey"
            | "thanks"
            | "thank you"
            | "bye"
            | "yes"
            | "no"
            | "status"
            | "ajuda"
            | "help"
            | "limpar"
            | "debug mem"
            | "clear"
            | "reset"
    )
}

/// Remove continuações de outra cláusula capturadas por regexes abertas.
fn clean_clause_tail(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    let lower = text.to_lowercase();

    let boundary_markers = [
        " e eu ",
        " e meu ",
        " e minha ",
        " e moro ",
        " e trabalho ",
        " e estudo ",
        " e gosto ",
        " e adoro ",
        " e amo ",
        " e sou ",
        " e tenho ",
        " e uso ",
        " e falo ",
        " e quero ",
        " e prefiro ",
        " e não ",
        " e nao ",
        " e odeio ",
        " mas ",
        " porque ",
        " pois ",
        " and i ",
        " and my ",
        " but ",
    ];

    let mut cut_at = text.len();
    for marker in boundary_markers {
        if let Some(pos) = lower.find(marker) {
            if pos < cut_at {
                cut_at = pos;
            }
        }
    }
    text.truncate(cut_at);
    clean_value(&text)
}

fn normalize_field(s: &str) -> String {
    s.trim().to_string()
}

fn clean_value(s: &str) -> String {
    s.trim()
        .trim_end_matches(|c: char| ".!?,;:".contains(c))
        .trim()
        .to_string()
}

fn is_valid_fact(fact: &ExtractedFact) -> bool {
    if fact.subject.is_empty() || fact.attribute.is_empty() || fact.value.is_empty() {
        return false;
    }

    let val_len = fact.value.chars().count();
    if !(2..=80).contains(&val_len) {
        return false;
    }

    let lower_subj = fact.subject.to_lowercase();
    let lower_attr = fact.attribute.to_lowercase();
    let lower_val = fact.value.to_lowercase();

    // Rejeita placeholders literais de template de prompt (ex.: [LEARN: Sujeito | atributo | valor])
    if matches!(
        lower_subj.as_str(),
        "sujeito" | "subject" | "pessoa" | "person"
    ) && matches!(
        lower_attr.as_str(),
        "atributo" | "attribute" | "chave" | "key"
    ) || matches!(
        lower_val.as_str(),
        "valor" | "value" | "exemplo" | "example" | "nada"
    ) {
        return false;
    }

    if lower_val.contains("<|")
        || lower_val.contains("</")
        || lower_val.contains("<think>")
        || lower_val.contains("[learn:")
    {
        return false;
    }

    let is_stopword = matches!(
        lower_val.as_str(),
        "isso"
            | "isto"
            | "aquilo"
            | "nada"
            | "tudo"
            | "algo"
            | "coisa"
            | "fazer"
            | "saber"
            | "ver"
            | "ter"
            | "ser"
            | "estar"
            | "ir"
            | "vir"
            | "aqui"
            | "ali"
            | "lá"
            | "la"
            | "hoje"
            | "amanhã"
            | "amanha"
            | "ontem"
            | "agora"
            | "sempre"
            | "nunca"
            | "sim"
            | "não"
            | "nao"
            | "ok"
            | "você"
            | "voce"
            | "ele"
            | "ela"
            | "eles"
            | "elas"
            | "muito"
            | "pouco"
            | "bem"
            | "mal"
            | "it"
            | "this"
            | "that"
            | "nothing"
            | "everything"
            | "something"
            | "thing"
            | "do"
            | "know"
            | "see"
            | "be"
            | "have"
            | "here"
            | "there"
            | "today"
            | "now"
            | "yes"
            | "no"
            | "you"
            | "he"
            | "she"
    );
    if is_stopword {
        return false;
    }

    if fact.attribute.eq_ignore_ascii_case("nome") {
        let invalid_name = matches!(
            lower_val.as_str(),
            "feliz"
                | "triste"
                | "cansado"
                | "cansada"
                | "legal"
                | "bom"
                | "boa"
                | "happy"
                | "sad"
                | "tired"
                | "good"
                | "fine"
                | "brazilian"
                | "brasileiro"
        );
        if invalid_name {
            return false;
        }
    }

    true
}

fn category_base_relevance(attribute: &str) -> f64 {
    let lower = attribute.to_lowercase();
    if lower == "nome" || lower == "name" {
        0.98
    } else if matches!(
        lower.as_str(),
        "localizacao" | "origem" | "profissao" | "trabalho" | "projeto" | "idade"
    ) {
        0.92
    } else if matches!(
        lower.as_str(),
        "estudo" | "preferencia" | "objetivo" | "idioma" | "pet" | "ferramenta"
    ) || lower.ends_with("_favorito")
    {
        0.88
    } else if matches!(lower.as_str(), "interesse" | "desinteresse") {
        0.84
    } else {
        0.80
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_filters_phatic_and_questions() {
        assert!(!RelevanceSystem::assess_user_input("oi").is_relevant);
        assert!(!RelevanceSystem::assess_user_input("Tudo bem?").is_relevant);
        assert!(!RelevanceSystem::assess_user_input("O que eu gosto de fazer?").is_relevant);
        assert!(!RelevanceSystem::assess_user_input("ok, obrigado!").is_relevant);

        assert!(RelevanceSystem::assess_user_input("Meu nome é Axel. Qual é o seu?").is_relevant);
        assert!(
            RelevanceSystem::assess_user_input("Trabalho como engenheiro de software.").is_relevant
        );
    }

    #[test]
    fn test_extract_rich_declarative_facts_and_relevance_scoring() {
        let input = "Oi! Meu nome é Axel e eu moro em Curitiba e trabalho como engenheiro. Tenho 28 anos. Qual é o seu nome?";
        let candidates = extract_declarative_facts(input);
        let facts = RelevanceSystem::score_and_filter(candidates);

        assert!(facts
            .iter()
            .any(|f| f.subject == "Usuario" && f.attribute == "nome" && f.value == "Axel"));
        assert!(facts.iter().any(|f| f.subject == "Usuario"
            && f.attribute == "localizacao"
            && f.value == "Curitiba"));
        assert!(facts.iter().any(|f| f.subject == "Usuario"
            && f.attribute == "profissao"
            && f.value == "engenheiro"));
        assert!(facts
            .iter()
            .any(|f| f.subject == "Usuario" && f.attribute == "idade" && f.value == "28 anos"));
        assert!(facts
            .iter()
            .all(|f| f.relevance >= MIN_FACT_RELEVANCE && f.relevance <= 1.0));
    }

    #[test]
    fn test_relevance_rejects_stopwords_bleeds_and_dedups() {
        let raw = vec![
            ExtractedFact {
                subject: "Usuario".into(),
                attribute: "interesse".into(),
                value: "isso".into(),
                relevance: 0.9,
            },
            ExtractedFact {
                subject: "Usuario".into(),
                attribute: "interesse".into(),
                value: "<|im_end|>".into(),
                relevance: 0.9,
            },
            ExtractedFact {
                subject: "Usuario".into(),
                attribute: "nome".into(),
                value: "feliz".into(),
                relevance: 0.9,
            },
            ExtractedFact {
                subject: "Sujeito".into(),
                attribute: "atributo".into(),
                value: "valor".into(),
                relevance: 0.9,
            },
            ExtractedFact {
                subject: "Usuario".into(),
                attribute: "interesse".into(),
                value: "Rust".into(),
                relevance: 0.70,
            },
            ExtractedFact {
                subject: "Usuario".into(),
                attribute: "interesse".into(),
                value: "Rust".into(),
                relevance: 0.95,
            },
        ];

        let filtered = RelevanceSystem::score_and_filter(raw);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].value, "Rust");
        assert!(filtered[0].relevance >= 0.88);
    }

    #[test]
    fn test_extract_preferences_projects_and_pets() {
        let input = "Não gosto de Java. Estou desenvolvendo o projeto ThrediaRs. Tenho um cachorro chamado Rex. Uso Linux.";
        let facts = RelevanceSystem::score_and_filter(extract_declarative_facts(input));

        assert!(facts
            .iter()
            .any(|f| f.attribute == "desinteresse" && f.value == "Java"));
        assert!(facts
            .iter()
            .any(|f| f.attribute == "projeto" && f.value == "o projeto ThrediaRs"));
        assert!(facts
            .iter()
            .any(|f| f.attribute == "pet" && f.value == "cachorro (Rex)"));
        assert!(facts
            .iter()
            .any(|f| f.attribute == "ferramenta" && f.value == "Linux"));
    }
}
