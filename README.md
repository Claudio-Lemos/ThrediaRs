# ThrediaRs

**ThrediaRs** é um assistente cognitivo local construído em Rust, projetado para rodar diretamente na máquina do usuário com foco em privacidade, baixa latência, memória persistente e personalidade dinâmica.

Inspirado em uma arquitetura modular dividida entre **Cérebro**, **Corpo** e **Telemetria**, o assistente combina modelos de linguagem locais, busca semântica textual (RAG), estado emocional contínuo e uma interface gráfica desktop moderna.

---

## Arquitetura do Projeto

O projeto é organizado como um workspace modular, separando responsabilidades cognitivas, de memória, sensoriais e de diagnóstico:

- **Núcleo e Interface (core)**: orquestra o fluxo de mensagens, comandos instantâneos, interface gráfica desktop, internacionalização e salvamento automático de configurações.
- **Cérebro (BRAIN / brain)**: reúne o motor de inferência de linguagem (Sistema 2), a Amígdala (Sistema 1), o sistema de relevância para extração de fatos, o estado emocional VAD e o compositor de prompts para múltiplas famílias de modelos de IA.
- **Memória (BRAIN / memory)**: gerencia o banco de dados local persistente com histórico de conversa, fatos estruturados e busca textual rápida para contexto (RAG).
- **Voz (BODY / speech)**: gerencia a saída de áudio em thread isolada para síntese de fala (TTS), com controle de ativação, seleção de modelo de voz e interrupção imediata.
- **Corpo e Ponte (BODY / body e BODY / bridge)**: prepara reflexos comportamentais baseados em emoções e a base de comunicação para futuras integrações externas (como jogos).
- **Telemetria e Diagnóstico (evolve)**: monitora recursos de hardware em tempo real, mede latência de operações, mantém logs em memória e gera diagnósticos automáticos.

---

## Principais Funcionalidades

### Cognição em Dois Sistemas
- **Sistema 2 — Raciocínio (Brain)**: executa modelos locais no formato GGUF com resposta em streaming token a token, controle completo de amostragem (temperatura, contexto, penalidades e limites de tokens), troca de modelo em tempo real (hot swap) e alternância entre CPU e GPU.
- **Sistema 1 — Instinto (Amígdala)**: analisa sentimento e excitação em tempo real por caminho rápido léxico e utiliza um modelo leve (SLM) em conjunto com um **Sistema de Relevância** para decidir quais informações da conversa do usuário merecem ser memorizadas.

### Memória Persistente e RAG
- Armazena histórico de conversas e fatos estruturados no formato **Sujeito, Atributo e Valor** com nível de confiança.
- Recupera automaticamente fatos relevantes antes de cada resposta para enriquecer o contexto da IA.
- Filtra ruídos, perguntas puras e vazamentos de marcações internas antes de gravar na memória.
- Sincroniza automaticamente referências recentes quando o nome da assistente é alterado.
- Permite visualizar, buscar, filtrar, adicionar, editar e remover fatos diretamente pela interface.

### Estado Emocional VAD e Homeostase
- Mantém um vetor emocional contínuo baseado em **Valência**, **Excitação** e **Dominância** (modelo VAD), que reage ao tom das mensagens do usuário e se traduz em descrições humanas de humor no prompt e na interface.
- Executa rotina periódica de homeostase que ajusta o humor gradualmente e reage ao estresse térmico ou de uso elevado de CPU.

### Interface Gráfica Desktop
- **Chat**: interface escura inspirada em mensageiros modernos, com indicador de humor, streaming ao vivo e opção de exibir ou ocultar blocos internos de raciocínio.
- **Aba Cérebro**: seleção e troca dinâmica de modelos LLM e SLM, organização automática na pasta de modelos, escolha de backend (CPU/GPU), ajustes do sampler, barras de humor VAD ao vivo e último veredito da Amígdala.
- **Aba Memória**: visualização de fatos e histórico, busca textual, sub-aba recolhível com **grade 3x3 scrollável de categorias por sujeito**, criação/edição manual de fatos e botão **Abrir SQL** para inspecionar o banco diretamente em um aplicativo compatível.
- **Aba Sistema**: seletor de idioma em tempo real, **editor de System Prompt com salvamento automático**, dropdown com templates padrão de diversas famílias de IA (ChatML/Qwen/SmolLM, Llama 3/3.2, TinyLlama/Zephyr, Gemma 2/3 e Mistral/Mixtral) e prompts customizados salvos pelo usuário, gráficos em tempo real de CPU e RAM, métricas de latência, Modo Economia de Energia e geração de relatório de diagnóstico (**The Blind Vault / Modo Sonho**).
- **Aba Voz**: ativação de TTS, seleção de arquivo de voz e interrupção imediata da fala.
- **Aba Logs**: histórico filtrável de eventos internos e diagnósticos do sistema.

### Internacionalização e Conveniência
- **Português (BR) e Inglês**: troca imediata de idioma na interface, nos comandos internos, nos humores VAD e nos templates padrão de prompt.
- **Salvamento Automático**: todas as preferências (modelos, sampler, prompt, idioma, voz e modo de energia) são salvas silenciosamente em arquivo JSON sempre que alteradas.
- **Comandos Instantâneos**: atalhos de latência zero para consultar status do sistema, listar memórias, limpar histórico/cache e exibir ajuda.

---

## Tecnologias Utilizadas

- **Linguagem e Build**: Rust (Cargo Workspace) com otimizações de compilação e alocador de memória **MiMalloc**.
- **Inferência Local de IA**: **llama-cpp-2** (binding nativo para llama.cpp com suporte a modelos GGUF e KV Cache quantizado) e **self_cell** para gerenciamento seguro de contexto.
- **Interface Gráfica**: **eframe** e **egui** para renderização nativa multiplataforma, além de **rfd** para diálogos nativos de seleção de arquivos.
- **Persistência e Busca**: **SQLite** via **SQLx** (em modo WAL) com extensão **FTS5** para busca textual rápida.
- **Assincronismo e Concorrência**: **Tokio** para pipeline assíncrono e **parking_lot** para sincronização de estado de baixa sobrecarga.
- **Processamento Léxico e Regras**: **memchr**, **rust-stemmers** e **regex** para análise rápida de texto, sanitização e despacho de comandos.
- **Áudio**: **Rodio** para reprodução persistente em thread dedicada.
- **Internacionalização**: **rust-i18n** com dicionários YAML em Português (BR) e Inglês.
- **Telemetria e Serialização**: **sysinfo**, **tracing**, **tracing-subscriber**, **chrono**, **serde** e **serde_json**.

---

## Roadmap

Entre as evoluções planejadas para o projeto estão:
- Síntese de voz neural local (Piper ONNX), reconhecimento de voz (Whisper com VAD) e ativação por palavra-chave (Wake Word).
- Identificação de múltiplos usuários, arquivamento de históricos antigos e memória espacial.
- Integração com ambientes externos e jogos (como ponte bidirecional para Minecraft) e futuras adaptações para dispositivos móveis.

---

## Estado do Projeto

O **ThrediaRs** está em desenvolvimento ativo em versão alpha. O histórico detalhado de alterações e correções está disponível em **CHANGELOG.md**.

