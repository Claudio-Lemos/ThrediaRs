# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

## [0.1.0-alpha.29] - 2026-11-19
### Adicionado
- **Editor de Prompt, Presets de Modelos de IA e Prompts Customizados na aba Sistema**:
  - Edição em tempo real do System Prompt com suporte a placeholders (`{name}`, `{role}`, `{mood}`) e salvamento automático em `settings.json`.
  - Dropdown com templates padrão de diversos modelos de IA (ChatML/Qwen/SmolLM, Llama 3/3.2, TinyLlama/Zephyr, Gemma 2/3 e Mistral/Mixtral) e suporte a salvar, atualizar e excluir prompts customizados.
  - Compatibilidade completa com Português (BR) e Inglês (`pt-BR` / `en`), adaptando automaticamente os templates padrão ao trocar o idioma.
- **Extração de Fatos com SLM da Amígdala e Sistema de Relevância**:
  - Extração de fatos passando pela SLM da Amígdala e pelo `RelevanceSystem` (avaliação de relevância, filtragem de ruído/stopwords/placeholders e persistência com confiança real no SQLite).
- **Melhorias na Aba Memória**:
  - Categorias de sujeitos exibidas como retângulos em grade 3x3 scrollável dentro de uma sub-aba recolhível para filtrar fatos rapidamente.
  - Botão **Abrir SQL** (*Open SQL*) para abrir o banco `data/memory.db` diretamente no aplicativo padrão do sistema, com aviso ao passar o mouse caso não haja programa compatível instalado.

## [0.1.0-alpha.28] - 2026-11-19
### Adicionado
- Suporte a idiomas (Português BR e Inglês) com seletor na aba Sistema e troca imediata em tempo real.
- Comandos internos, prompts do sistema e extração de fatos adaptados ao idioma selecionado.
- Atualização automática do nome do assistente reconhecendo outros idiomas.

### Corrigido
- Ajustada a configuração de compilação para permitir cross-compilation para Windows (x86_64) e Linux ARM64 (aarch64).

## [0.1.0-alpha.27] - 2026-11-19
### Alterado
- Salvamento automático e silencioso das configurações ao alterar opções na interface, removendo a necessidade de botões manuais de salvar/exportar/importar.

## [0.1.0-alpha.26] - 2026-11-19
### Adicionado
- Persistência de configurações da interface (modelos, backend, sampler, temperatura, contexto, modo economia e voz) em arquivo JSON.

## [0.1.0-alpha.25] - 2026-11-19
### Corrigido
- Removidos caracteres residuais (`<`, `>`, `|` e `</`) no final das mensagens salvas no histórico e exibidas no chat, preservando comparações normais no meio das frases.

## [0.1.0-alpha.24] - 2026-11-19
### Corrigido
- Melhorada a limpeza de marcações residuais ao final das respostas geradas pelo modelo.

## [0.1.0-alpha.23] - 2026-11-19
### Corrigido
- Adicionada sanitização do histórico para evitar que tags internas ou vazamentos de template do modelo sejam salvos no banco de dados.

## [0.1.0-alpha.22] - 2026-11-19
### Corrigido
- Adicionada rolagem vertical na aba Cérebro para impedir corte de controles em janelas menores.

## [0.1.0-alpha.21] - 2026-11-19
### Adicionado
- Controles completos de amostragem (Top-K, Top-P, Min-P, penalidades e limite de tokens) na aba Cérebro, com botão para restaurar padrões.

## [0.1.0-alpha.20] - 2026-11-19
### Adicionado
- Gráficos em tempo real de CPU e RAM, métricas de latência por operação e painel de diagnóstico na aba Sistema.
- Relatório automático de diagnóstico em modo isolado quando erros são registrados.

## [0.1.0-alpha.19] - 2026-11-19
### Corrigido
- Corrigido artefato `>` no fim das respostas com modelos Llama 3.2 e adicionados marcadores de parada nativos.
- Removidos avisos de compilação nos manifestos do projeto.

## [0.1.0-alpha.18] - 2026-11-19
### Adicionado
- Modo de teste conversacional ponta a ponta com banco em memória.

### Corrigido
- Evitados falsos positivos na extração de fatos quando o usuário faz perguntas.
- Reduzida a verbosidade de logs internos do motor de inferência.

## [0.1.0-alpha.17] - 2026-11-19
### Corrigido
- Corrigida a tokenização do formato ChatML para evitar duplicação de token inicial.
- Refinados os parâmetros padrão de amostragem para reduzir alucinações.

## [0.1.0-alpha.16] - 2026-11-19
### Adicionado
- Criação automática da pasta `modelos/` junto ao executável, com detecção e carregamento automático de modelos `.gguf`.
- Opção na aba Cérebro para mover o modelo selecionado para a pasta `modelos/`.

### Corrigido
- Limpeza automática de mensagens antigas de fallback na inicialização.

## [0.1.0-alpha.15] - 2026-11-19
### Corrigido
- Corrigida a detecção de fim de turno e penalidades de repetição para evitar loops em modelos pequenos.
- Corrigida a ordem de consulta do histórico para evitar duplicação da mensagem atual no prompt.

## [0.1.0-alpha.14] - 2026-11-19
### Alterado
- Inferência nativa via `llama.cpp` habilitada por padrão na compilação.

### Corrigido
- Atualizada a integração com a versão mais recente do motor de inferência.

## [0.1.0-alpha.13] - 2026-11-19
### Adicionado
- Motor de inferência local para modelos GGUF com suporte a streaming token a token e aceleração por GPU.

## [0.1.0-alpha.12] - 2026-11-19
### Adicionado
- Seleção e troca em tempo real (Hot Swap) do modelo auxiliar (SLM/Amígdala) na aba Cérebro.
- Extração híbrida de fatos combinando análise rápida e marcações do modelo principal.

## [0.1.0-alpha.11] - 2026-11-19
### Corrigido
- Travada a largura fixa da barra lateral e adicionada quebra de linha automática na aba Logs.

## [0.1.0-alpha.10] - 2026-11-19
### Adicionado
- Propagação automática do nome do assistente na interface e no histórico recente quando alterado na memória.

## [0.1.0-alpha.9] - 2026-11-19
### Adicionado
- Nova aba Memória na interface para visualizar, filtrar, adicionar, editar e excluir fatos e mensagens do histórico.

## [0.1.0-alpha.8] - 2026-11-19
### Alterado
- Simplificada a seleção de modelos `.gguf` utilizando exclusivamente o diálogo nativo do sistema operacional.

## [0.1.0-alpha.7] - 2026-11-19
### Adicionado
- Diálogos nativos não-bloqueantes para seleção de modelos de linguagem e voz.

## [0.1.0-alpha.6] - 2026-11-19
### Adicionado
- Seleção dinâmica de modelos `.gguf`, alternância entre CPU e GPU e troca de modelo em tempo real (Hot Swap) sem reiniciar o programa.

## [0.1.0-alpha.5] - 2026-11-19
### Adicionado
- Interface gráfica completa em modo escuro com chat em tempo real, cabeçalho com indicador de humor e painel lateral com abas Cérebro, Sistema, Voz e Logs.

## [0.1.0-alpha.4] - 2026-11-19
### Adicionado
- Módulo de saída de áudio/voz em thread dedicada com suporte a parada imediata.
- Comandos rápidos integrados no chat (status do sistema, listar memória, limpar histórico e ajuda).
- Pipeline completo de processamento de mensagens com streaming de resposta.

## [0.1.0-alpha.3] - 2026-11-19
### Adicionado
- Sistema emocional baseado no modelo VAD (Valência, Excitação e Dominância) com homeostase automática.
- Compositor de prompts e módulo de análise rápida de sentimento e fatos.

## [0.1.0-alpha.2] - 2026-11-19
### Adicionado
- Memória persistente em SQLite com histórico de conversas, tabela de fatos e busca textual (FTS5).

## [0.1.0-alpha.1] - 2026-11-19
### Adicionado
- Estrutura inicial do projeto em workspace Rust.
- Sistema de telemetria e monitoramento de hardware (CPU, RAM e temperatura).
- Janela nativa inicial da aplicação.

