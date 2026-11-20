use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

/// Remove sangramentos de template de IA (`<|im_end|`, `<|im_start|>`, `<|eot_id|>`, `</s>`,
/// `<end_of_turn>`, `<think>...</think>`, `[LEARN:...]`) antes de persistir no histórico.
pub fn sanitize_history_content(raw: &str) -> String {
    static TURN_BLEED_RE: OnceLock<Regex> = OnceLock::new();
    static THINK_COMPLETE_RE: OnceLock<Regex> = OnceLock::new();
    static THINK_OPEN_RE: OnceLock<Regex> = OnceLock::new();
    static SPECIAL_COMPLETE_RE: OnceLock<Regex> = OnceLock::new();
    static SPECIAL_INCOMPLETE_RE: OnceLock<Regex> = OnceLock::new();
    static TEMPLATE_TOKENS_RE: OnceLock<Regex> = OnceLock::new();
    static LEARN_TAG_RE: OnceLock<Regex> = OnceLock::new();
    static STRAY_CLOSE_RE: OnceLock<Regex> = OnceLock::new();

    let turn_bleed = TURN_BLEED_RE.get_or_init(|| {
        Regex::new(r"(?is)(?:<\s*\|\s*im_start\|?|<\s*\|\s*im_end\|?|<\s*\|\s*eot_id\|?|<\s*\|\s*end_of_text\|?|<\s*\|\s*endoftext\|?|<\s*\|\s*start_header_id\|?|<\s*\|\s*end_header_id\|?|<end_of_turn>|<start_of_turn>).*$").unwrap()
    });
    let think_complete = THINK_COMPLETE_RE.get_or_init(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    let think_open = THINK_OPEN_RE.get_or_init(|| Regex::new(r"(?s)<think>.*$").unwrap());
    let special_complete = SPECIAL_COMPLETE_RE.get_or_init(|| Regex::new(r"<\s*\|[^|\n]*\|\s*>").unwrap());
    let special_incomplete = SPECIAL_INCOMPLETE_RE.get_or_init(|| Regex::new(r"<\s*\|[^\n>]*$").unwrap());
    let template_tokens = TEMPLATE_TOKENS_RE
        .get_or_init(|| Regex::new(r"(?i)(?:</?s>|\[/?INST\])").unwrap());
    let learn_tag = LEARN_TAG_RE.get_or_init(|| Regex::new(r"(?i)\[LEARN:[^\]]*\]").unwrap());
    let stray_close = STRAY_CLOSE_RE
        .get_or_init(|| Regex::new(r"(?:</?|<\s*\|?|\|\s*>?|\s+>|\n>)\s*$").unwrap());

    let mut text = raw.to_string();
    text = think_complete.replace_all(&text, "").into_owned();
    text = think_open.replace_all(&text, "").into_owned();
    text = turn_bleed.replace_all(&text, "").into_owned();
    text = special_complete.replace_all(&text, "").into_owned();
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fact {
    pub id: i64,
    pub subject: String,
    pub attribute: String,
    pub value: String,
    pub confidence: f64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Clone)]
pub struct Memory {
    pool: SqlitePool,
}

impl Memory {
    /// Retorna o caminho padrão do banco SQLite (`data/memory.db` relativo ao executável atual).
    pub fn default_db_path() -> PathBuf {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        exe_dir.join("data").join("memory.db")
    }

    /// Abre o banco em `data/memory.db` relativo ao executável atual.
    pub async fn open_default() -> Result<Self> {
        let db_path = Self::default_db_path();
        if let Some(db_dir) = db_path.parent() {
            std::fs::create_dir_all(db_dir)
                .with_context(|| format!("Falha ao criar diretório {:?}", db_dir))?;
        }
        Self::open_path(&db_path).await
    }

    pub async fn open_path(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .with_context(|| format!("Falha ao abrir SQLite em {:?}", path))?;

        let mem = Self { pool };
        mem.init_schema().await?;
        Ok(mem)
    }

    /// Abre banco SQLite em memória (`sqlite::memory:`) para testes unitários.
    pub async fn open_in_memory() -> Result<Self> {
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")?
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await?;

        let mem = Self { pool };
        mem.init_schema().await?;
        Ok(mem)
    }

    async fn init_schema(&self) -> Result<()> {
        sqlx::query("PRAGMA journal_mode = WAL;")
            .execute(&self.pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL;")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique
            ON facts (subject, attribute, value);
            "#,
        )
        .execute(&self.pool)
        .await?;

        // FTS5 contentless vinculada à tabela facts
        sqlx::query(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                subject,
                attribute,
                value,
                content='facts',
                content_rowid='id'
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, subject, attribute, value)
                VALUES (new.id, new.subject, new.attribute, new.value);
            END;
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, subject, attribute, value)
                VALUES ('delete', old.id, old.subject, old.attribute, old.value);
            END;
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, subject, attribute, value)
                VALUES ('delete', old.id, old.subject, old.attribute, old.value);
                INSERT INTO facts_fts(rowid, subject, attribute, value)
                VALUES (new.id, new.subject, new.attribute, new.value);
            END;
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Rebuild do índice FTS5 no boot para garantir consistência
        sqlx::query("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild');")
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn learn_fact(
        &self,
        subject: &str,
        attribute: &str,
        value: &str,
        confidence: f64,
    ) -> Result<()> {
        let subject = subject.trim();
        let attribute = attribute.trim();
        let value = value.trim();
        if subject.is_empty() || attribute.is_empty() || value.is_empty() {
            return Ok(());
        }

        sqlx::query(
            r#"
            INSERT INTO facts (subject, attribute, value, confidence)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(subject, attribute, value)
            DO UPDATE SET confidence = MAX(confidence, excluded.confidence),
                          created_at = datetime('now');
            "#,
        )
        .bind(subject)
        .bind(attribute)
        .bind(value)
        .bind(confidence.clamp(0.0, 1.0))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Retorna o valor mais recente (MAX(id)) por atributo para um sujeito (ex.: identidade do sistema).
    pub async fn get_identity(&self, subject: &str) -> Result<HashMap<String, String>> {
        let rows = sqlx::query(
            r#"
            SELECT f.attribute, f.value
            FROM facts f
            INNER JOIN (
                SELECT attribute, MAX(id) AS max_id
                FROM facts
                WHERE subject = ?1
                GROUP BY attribute
            ) latest ON f.id = latest.max_id
            ORDER BY f.attribute ASC;
            "#,
        )
        .bind(subject)
        .fetch_all(&self.pool)
        .await?;

        let mut map = HashMap::new();
        for row in rows {
            let attr: String = row.try_get("attribute")?;
            let val: String = row.try_get("value")?;
            map.insert(attr, val);
        }
        Ok(map)
    }

    pub async fn get_all_facts(&self) -> Result<Vec<Fact>> {
        let rows = sqlx::query(
            r#"
            SELECT id, subject, attribute, value, confidence, created_at
            FROM facts
            ORDER BY id DESC;
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok(Fact {
                    id: r.try_get("id")?,
                    subject: r.try_get("subject")?,
                    attribute: r.try_get("attribute")?,
                    value: r.try_get("value")?,
                    confidence: r.try_get("confidence")?,
                    created_at: r.try_get("created_at")?,
                })
            })
            .collect()
    }

    /// Busca memórias via FTS5 com sanitização alfanumérica + query OR.
    pub async fn search_memories(&self, query: &str, limit: i64) -> Result<Vec<Fact>> {
        let tokens: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.chars().count() >= 2)
            .map(|t| format!("{}*", t.to_lowercase()))
            .collect();

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let fts_query = tokens.join(" OR ");

        let rows = sqlx::query(
            r#"
            SELECT f.id, f.subject, f.attribute, f.value, f.confidence, f.created_at
            FROM facts_fts
            JOIN facts f ON f.id = facts_fts.rowid
            WHERE facts_fts MATCH ?1
            ORDER BY bm25(facts_fts) ASC, f.id DESC
            LIMIT ?2;
            "#,
        )
        .bind(&fts_query)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok(Fact {
                    id: r.try_get("id")?,
                    subject: r.try_get("subject")?,
                    attribute: r.try_get("attribute")?,
                    value: r.try_get("value")?,
                    confidence: r.try_get("confidence")?,
                    created_at: r.try_get("created_at")?,
                })
            })
            .collect()
    }

    pub async fn add_message(&self, role: &str, content: &str) -> Result<()> {
        let clean = sanitize_history_content(content);
        if clean.is_empty() {
            return Ok(());
        }
        sqlx::query("INSERT INTO chat_history (role, content) VALUES (?1, ?2);")
            .bind(role)
            .bind(&clean)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Retorna histórico recente formatado em ChatML (`<|im_start|>role\ncontent<|im_end|>`).
    pub async fn get_recent_chat(&self, limit: i64) -> Result<String> {
        let msgs = self.get_recent_messages(limit).await?;
        let mut out = String::new();
        for m in msgs {
            out.push_str("<|im_start|>");
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        }
        Ok(out)
    }

    pub async fn get_recent_messages(&self, limit: i64) -> Result<Vec<ChatMessage>> {
        let rows = sqlx::query(
            r#"
            SELECT id, role, content, created_at
            FROM (
                SELECT id, role, content, created_at
                FROM chat_history
                ORDER BY id DESC
                LIMIT ?1
            )
            ORDER BY id ASC;
            "#,
        )
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                Ok(ChatMessage {
                    id: r.try_get("id")?,
                    role: r.try_get("role")?,
                    content: r.try_get("content")?,
                    created_at: r.try_get("created_at")?,
                })
            })
            .collect()
    }

    pub async fn clear_chat_history(&self) -> Result<()> {
        sqlx::query("DELETE FROM chat_history;")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn update_fact(
        &self,
        id: i64,
        subject: &str,
        attribute: &str,
        value: &str,
        confidence: f64,
    ) -> Result<()> {
        let subject = subject.trim();
        let attribute = attribute.trim();
        let value = value.trim();
        if subject.is_empty() || attribute.is_empty() || value.is_empty() {
            return Ok(());
        }
        sqlx::query(
            r#"
            UPDATE facts
            SET subject = ?1, attribute = ?2, value = ?3, confidence = ?4
            WHERE id = ?5;
            "#,
        )
        .bind(subject)
        .bind(attribute)
        .bind(value)
        .bind(confidence.clamp(0.0, 1.0))
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_fact(&self, id: i64) -> Result<()> {
        sqlx::query("DELETE FROM facts WHERE id = ?1;")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn delete_chat_message(&self, id: i64) -> Result<()> {
        sqlx::query("DELETE FROM chat_history WHERE id = ?1;")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Remove mensagens antigas de aviso de protótipo/fallback para não poluir o contexto do LLM.
    pub async fn purge_stale_prototype_messages(&self) -> Result<u64> {
        let res = sqlx::query(
            r#"
            DELETE FROM chat_history
            WHERE content LIKE '%compilado sem o motor%'
               OR content LIKE '%requer `cmake`%'
               OR content LIKE '%Quiz: Qual é a lista de modelos%';
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected())
    }

    /// Substitui referências ao nome antigo do assistente pelo novo nome nas `recent_limit`
    /// mensagens mais recentes do `chat_history`, usando fronteira de palavra (`\b`).
    pub async fn replace_recent_name_references(
        &self,
        old_name: &str,
        new_name: &str,
        recent_limit: i64,
    ) -> Result<usize> {
        let old_trimmed = old_name.trim();
        let new_trimmed = new_name.trim();
        if old_trimmed.is_empty()
            || new_trimmed.is_empty()
            || old_trimmed.eq_ignore_ascii_case(new_trimmed)
        {
            return Ok(0);
        }

        let pattern = format!(r"(?i)\b{}\b", regex::escape(old_trimmed));
        let re = Regex::new(&pattern)?;

        let rows = sqlx::query(
            r#"
            SELECT id, content
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?1;
            "#,
        )
        .bind(recent_limit.max(1))
        .fetch_all(&self.pool)
        .await?;

        let mut updated = 0usize;
        for row in rows {
            let id: i64 = row.try_get("id")?;
            let content: String = row.try_get("content")?;
            if re.is_match(&content) {
                let replaced = re.replace_all(&content, new_trimmed).into_owned();
                if replaced != content {
                    sqlx::query("UPDATE chat_history SET content = ?1 WHERE id = ?2;")
                        .bind(&replaced)
                        .bind(id)
                        .execute(&self.pool)
                        .await?;
                    updated += 1;
                }
            }
        }
        Ok(updated)
    }
}

/// Retorna `true` se o par `(subject, attribute)` representa o nome do assistente
/// em qualquer um dos idiomas suportados (PT, EN, ES, FR).
pub fn is_assistant_name_fact(subject: &str, attribute: &str) -> bool {
    let subj = subject.trim().to_ascii_lowercase();
    let attr = attribute.trim().to_ascii_lowercase();
    let is_assistant_subject = matches!(
        subj.as_str(),
        "assistente" | "assistant" | "asistente" | "ai" | "ia"
    );
    let is_name_attr = matches!(
        attr.as_str(),
        "nome" | "name" | "nombre" | "nom" | "nome_assistente" | "assistant_name"
    );
    is_assistant_subject && is_name_attr
}

/// Macro leve que detecta alteração no atributo de nome do assistente (em PT, EN, ES, FR)
/// e propaga automaticamente o novo nome nas referências mais recentes do histórico.
#[macro_export]
macro_rules! sync_assistant_name_if_changed {
    ($mem:expr, $subject:expr, $attribute:expr, $old_name:expr, $new_name:expr, $limit:expr) => {{
        if $crate::is_assistant_name_fact($subject, $attribute) {
            $mem.replace_recent_name_references($old_name, $new_name, $limit)
                .await
        } else {
            Ok(0)
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_identity_and_facts_eav() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.learn_fact("Assistente", "nome", "Theia-v0", 0.9)
            .await
            .unwrap();
        mem.learn_fact("Assistente", "nome", "Theia", 1.0)
            .await
            .unwrap();
        mem.learn_fact("Assistente", "idioma", "PT-BR", 1.0)
            .await
            .unwrap();

        let id = mem.get_identity("Assistente").await.unwrap();
        assert_eq!(id.get("nome").map(String::as_str), Some("Theia"));
        assert_eq!(id.get("idioma").map(String::as_str), Some("PT-BR"));

        let all = mem.get_all_facts().await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_fts5_search_memories() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.learn_fact("Usuario", "linguagem_favorita", "Rust", 1.0)
            .await
            .unwrap();
        mem.learn_fact("Usuario", "jogo", "Minecraft", 0.95)
            .await
            .unwrap();

        let hits = mem
            .search_memories("Qual minha linguagem Rust???", 5)
            .await
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].value, "Rust");
    }

    #[tokio::test]
    async fn test_chat_history_chatml_and_clear() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.add_message("user", "Olá!").await.unwrap();
        mem.add_message("assistant", "Oi! Tudo bem?").await.unwrap();

        let chatml = mem.get_recent_chat(10).await.unwrap();
        assert!(chatml.contains("<|im_start|>user\nOlá!<|im_end|>"));
        assert!(chatml.contains("<|im_start|>assistant\nOi! Tudo bem?<|im_end|>"));

        mem.clear_chat_history().await.unwrap();
        let after = mem.get_recent_chat(10).await.unwrap();
        assert!(after.is_empty());
    }

    #[tokio::test]
    async fn test_update_and_delete_fact_and_message() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.learn_fact("Usuario", "cidade", "Curitiba", 0.8)
            .await
            .unwrap();
        let facts = mem.get_all_facts().await.unwrap();
        assert_eq!(facts.len(), 1);
        let fid = facts[0].id;

        mem.update_fact(fid, "Usuario", "cidade", "Florianopolis", 0.95)
            .await
            .unwrap();
        let hits = mem.search_memories("Florianopolis", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].value, "Florianopolis");

        mem.delete_fact(fid).await.unwrap();
        assert!(mem.get_all_facts().await.unwrap().is_empty());
        assert!(mem.search_memories("Florianopolis", 5).await.unwrap().is_empty());

        mem.add_message("user", "mensagem 1").await.unwrap();
        let msgs = mem.get_recent_messages(10).await.unwrap();
        assert_eq!(msgs.len(), 1);
        mem.delete_chat_message(msgs[0].id).await.unwrap();
        assert!(mem.get_recent_messages(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_sync_assistant_name_macro_updates_recent_history() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.add_message("assistant", "Olá! Eu sou a Theia, sua assistente.")
            .await
            .unwrap();
        mem.add_message("user", "Theia, qual a sua missão diária?")
            .await
            .unwrap();

        let changed = sync_assistant_name_if_changed!(
            &mem,
            "Assistente",
            "nome",
            "Theia",
            "Luna",
            50
        )
        .unwrap();
        assert_eq!(changed, 2);

        let msgs = mem.get_recent_messages(10).await.unwrap();
        assert_eq!(msgs[0].content, "Olá! Eu sou a Luna, sua assistente.");
        assert_eq!(msgs[1].content, "Luna, qual a sua missão diária?");

        let changed_en = sync_assistant_name_if_changed!(
            &mem,
            "Assistant",
            "name",
            "Luna",
            "Nova",
            50
        )
        .unwrap();
        assert_eq!(changed_en, 2);

        let changed_es = sync_assistant_name_if_changed!(
            &mem,
            "Asistente",
            "nombre",
            "Nova",
            "Iris",
            50
        )
        .unwrap();
        assert_eq!(changed_es, 2);

        let noop = sync_assistant_name_if_changed!(
            &mem,
            "Usuario",
            "nome",
            "Iris",
            "Outro",
            50
        )
        .unwrap();
        assert_eq!(noop, 0);
    }

    #[tokio::test]
    async fn test_sanitize_history_prevents_ai_bleeds() {
        let mem = Memory::open_in_memory().await.unwrap();
        mem.add_message("assistant", "Olá!<|im_end|")
            .await
            .unwrap();
        mem.add_message("assistant", "Resposta boa<|im_start|>user\nFalso")
            .await
            .unwrap();
        mem.add_message("assistant", "Certo<|eot_id")
            .await
            .unwrap();
        mem.add_message("assistant", "Teste sem barra<")
            .await
            .unwrap();
        mem.add_message("assistant", "Outro teste <")
            .await
            .unwrap();
        mem.add_message("assistant", "Sem espaco>")
            .await
            .unwrap();
        mem.add_message("assistant", "Multiplos > > |")
            .await
            .unwrap();
        mem.add_message("assistant", "<think>só pensamento</think><|im_end|")
            .await
            .unwrap();

        let msgs = mem.get_recent_messages(10).await.unwrap();
        assert_eq!(msgs.len(), 7);
        assert_eq!(msgs[0].content, "Olá!");
        assert_eq!(msgs[1].content, "Resposta boa");
        assert_eq!(msgs[2].content, "Certo");
        assert_eq!(msgs[3].content, "Teste sem barra");
        assert_eq!(msgs[4].content, "Outro teste");
        assert_eq!(msgs[5].content, "Sem espaco");
        assert_eq!(msgs[6].content, "Multiplos");
    }
}
