# HW RAG MCP · History Essay Knowledge Base. Сделан Дмитрий Каневский

Проект выполняет домашнее задание по теме **RAG и протоколы взаимодействия ИИ-агентов**.

Итоговая схема проекта:

```text
нормализованные JSON-документы
  → enriched chunks
  → metadata + embedding_text
  → локальный Qdrant index
  → MCP-сервер с поисковыми tools
  → LangChain agent
  → top-k темы исторических эссе с document_id, chunk_id и source
```

Корпус проекта — темы исторических эссе школьных олимпиад по истории. Исходные PDF-документы заранее были преобразованы в нормализованные JSON-файлы: из них оставлены только темы исторических эссе и полезные метаданные. Далее проект строит RAG-индекс и демонстрирует поиск через MCP-инструменты и агента.

## Что сделано

В проекте реализовано:

1. Подготовка корпуса из JSON-файлов.
2. Создание enriched chunks с устойчивыми идентификаторами `document_id` и `chunk_id`.
3. Формирование двух представлений chunk:
   - `embedding_text` — компактный текст для embeddings;
   - `display_text` / metadata — данные для вывода и проверки результата.
4. Индексация chunks в локальный Qdrant.
5. Поиск по Qdrant:
   - semantic search;
   - поиск по автору цитаты;
   - поиск по типу авторской интерпретации;
   - metadata-aware reranking.
6. MCP-сервер поверх поискового сервиса.
7. Подключение MCP tools к LangChain agent через `langchain-mcp-adapters`.
8. Демонстрация agent/tool вызовов в notebook.
9. Eval-набор из 20 запросов и ручной анализ качества поиска.

## Важное замечание про MCP-сервер

В задании предлагается использовать MCP-сервер для базы знаний, поискового сервиса или векторного хранилища. В этом проекте используется **FastMCP** как готовый MCP-фреймворк, а сервер `hw_rag_mcp.mcp_server` является тонким adapter layer поверх собственного Qdrant-backed search service.

То есть MCP-сервер не занимается подготовкой корпуса и не реализует векторную базу сам. Он только публикует готовые поисковые сценарии как MCP tools:

- `semantic_search`;
- `search_by_quote_author`;
- `search_by_query_and_interpretation_type`.

Поиск, индексация и хранение embeddings выполняются через локальный Qdrant index. Такой вариант выбран, чтобы явно показать протокольную границу между агентом и поисковой системой.

## Структура проекта

```text
.
├── README.md
├── data
│   ├── chunks
│   │   ├── chunks.json
│   │   └── chunks.jsonl
│   ├── eval
│   │   ├── eval_queries.json
│   │   └── eval_results.json
│   ├── raw_json
│   │   ├── final_2015_v3.json
│   │   ├── ...
│   │   └── reg_2023_v3.json
│   └── vectorstore
│       └── qdrant
│           ├── collection/history_essay_chunks/storage.sqlite
│           └── meta.json
├── mcp_config.json
├── notebooks
│   ├── 1_ingest.ipynb
│   ├── 2_search_demo.ipynb
│   ├── 3_agent_demo.ipynb
│   └── eval_report.md
├── prompts
│   ├── pdf_to_history_essay_json_system_prompt_v3.md
│   └── user_prompt.md
├── pyproject.toml
├── src
│   └── hw_rag_mcp
│       ├── chunks
│       │   ├── build_chunks.py
│       │   └── schemas.py
│       ├── search_service
│       │   ├── models.py
│       │   └── service.py
│       ├── vectorstore
│       │   ├── embedding_model_port.py
│       │   ├── models.py
│       │   └── qdrant_indexer.py
│       ├── mcp_server.py
│       └── settings.py
└── uv.lock
```

## Основные директории

### `data/raw_json`

Нормализованные JSON-файлы по годам и этапам олимпиады. Каждый JSON соответствует одному исходному документу: региональному или заключительному этапу.

### `data/chunks`

Результат подготовки корпуса:

- `chunks.json` — человекочитаемый JSON со всеми enriched chunks;
- `chunks.jsonl` — построчный формат, удобный для последующей обработки.

### `data/vectorstore/qdrant`

Локальное persistent-хранилище Qdrant. В нём лежит коллекция `history_essay_chunks` с embedding vectors и payload metadata.

### `data/eval`

Eval-данные:

- `eval_queries.json` — 20 проверочных запросов;
- `eval_results.json` — ответы агента на эти запросы.

### `notebooks`

Основной путь проверки проекта. Тетрадки нужно запускать по порядку:

1. `1_ingest.ipynb`;
2. `2_search_demo.ipynb`;
3. `3_agent_demo.ipynb`.

Также здесь лежит `eval_report.md` — ручной анализ eval-прогона.

### `src/hw_rag_mcp/chunks`

Доменная логика chunks:

- Pydantic-схемы документов и тем;
- модель `EnrichedChunk`;
- построение `embedding_text`, `display_text`, `metadata`;
- `ChunkBuilder`, который читает JSON и сохраняет chunks.

### `src/hw_rag_mcp/vectorstore`

Слой работы с Qdrant:

- создание / открытие коллекции;
- индексация chunks;
- semantic search;
- metadata search;
- проверка количества points и параметров коллекции.

### `src/hw_rag_mcp/search_service`

Use-case слой над индексером. Здесь собраны сценарии поиска:

- обычный semantic search;
- поиск по автору цитаты;
- поиск по типу интерпретации;
- metadata-aware reranking.

### `src/hw_rag_mcp/mcp_server.py`

MCP-сервер проекта. Он публикует три tools:

1. `semantic_search(query)` — смысловой поиск по корпусу.
2. `search_by_quote_author(author)` — поиск по structured metadata автора цитаты.
3. `search_by_query_and_interpretation_type(query, interpretation_type)` — semantic search + усиление результатов по типу авторской интерпретации.

Tools возвращают компактный payload:

```json
{
  "document_id": "final_2015",
  "chunk_id": "final_2015_topic_001",
  "source": "final_2015.pdf",
  "author": "В.В. Мавродин",
  "content": "..."
}
```

### `src/hw_rag_mcp/settings.py`

Единая конфигурация проекта на `pydantic-settings`. Используется в notebooks, Qdrant indexer и MCP server.

## Установка

Проект рассчитан на запуск через `uv`.

```bash
uv sync
```

Если окружение ещё не создано, `uv` создаст `.venv` и установит зависимости из `pyproject.toml` / `uv.lock`.

## Переменные окружения

Нужно создать `.env` в корне проекта. Пример переменных:

```env
GIGACHAT_CREDENTIALS=...
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-2-Max
GIGACHAT_EMBEDDINGS_MODEL=EmbeddingsGigaR
GIGACHAT_VERIFY_SSL_CERTS=false

QDRANT_PATH=data/vectorstore/qdrant
QDRANT_COLLECTION=history_essay_chunks

LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Langfuse нужен только для демонстрации наблюдаемости в `3_agent_demo.ipynb`. Сам MCP-сервер не зависит от Langfuse.

## Как запускать проект

### Шаг 1. Ingest

Открыть и выполнить:

```text
notebooks/1_ingest.ipynb
```

Что происходит:

1. Загружаются нормализованные JSON-файлы из `data/raw_json`.
2. Каждый документ валидируется через Pydantic-схемы.
3. Каждая тема исторического эссе превращается в `EnrichedChunk`.
4. Сохраняются `chunks.json` и `chunks.jsonl`.
5. Создаются embeddings через GigaChatEmbeddings.
6. Chunks индексируются в локальный Qdrant.
7. Проверяется, что коллекция создана и содержит points.

Важно: локальный Qdrant блокирует директорию хранения. После завершения notebook нужно закрывать Qdrant client или перезапускать kernel перед запуском следующей тетрадки, если возникает lock error.

### Шаг 2. Search demo

Открыть и выполнить:

```text
notebooks/2_search_demo.ipynb
```

Что происходит:

1. Открывается уже созданный Qdrant index.
2. Проверяется обычный dense search.
3. Показываются ограничения dense search:
   - он может плохо находить конкретного автора цитаты;
   - он не всегда учитывает тип авторской позиции.
4. Добавляются metadata-based поиски:
   - по `quote_author`;
   - по `interpretation_type`.
5. Демонстрируется metadata-aware reranking.
6. Показывается пример, где negative assessment по Александру Невскому поднимает правильную тему выше.

### Шаг 3. Agent demo + MCP

Открыть и выполнить:

```text
notebooks/3_agent_demo.ipynb
```

Что происходит:

1. Проверяется подключение Langfuse.
2. Через `MultiServerMCPClient` запускается MCP-сервер по stdio.
3. MCP tools загружаются как LangChain tools.
4. Tools проверяются вручную без агента.
5. Создаётся LangChain agent.
6. Агент вызывает MCP tools по пользовательским запросам.
7. В Langfuse можно посмотреть LLM-вызовы, tool calls и ответы tools.
8. Создаётся eval-набор из 20 запросов.
9. Запросы прогоняются через агента.
10. Результаты сохраняются в `data/eval/eval_results.json`.

## MCP config

Пример конфигурации лежит в:

```text
mcp_config.json
```

Логика запуска MCP-сервера:

```bash
uv run python -m hw_rag_mcp.mcp_server
```

В notebook сервер обычно запускается не вручную, а через `langchain-mcp-adapters` и `MultiServerMCPClient`.

## Eval

Eval-набор лежит здесь:

```text
data/eval/eval_queries.json
```

Результаты прогона агента:

```text
data/eval/eval_results.json
```

Ручной отчёт:

```text
notebooks/eval_report.md
```

Eval содержит 20 запросов разных типов:

- обычные смысловые запросы;
- поиск по автору цитаты;
- поиск по типу авторской интерпретации;
- широкие запросы;
- запросы с опечатками;
- no-answer запросы, где в корпусе нет хорошего ответа;
- metadata-sensitive запросы.

Итог ручной оценки:

```text
Good:    16
Partial: 4
Bad:     0
```

Главный вывод eval: система в целом успешно возвращает релевантные темы с `document_id`, `chunk_id`, `source` и текстом. Наиболее слабое место — сложные причинно-следственные запросы, например про причины и предпосылки реформ Петра I.

## Соответствие требованиям задания

| Требование задания | Как выполнено в проекте |
|---|---|
| Подготовить корпус документов | Использован корпус тем исторических эссе из JSON-файлов в `data/raw_json`. |
| Разбить документы на фрагменты | Каждая тема эссе превращается в отдельный `EnrichedChunk`. |
| Сохранить metadata | Для каждого chunk сохраняются `document_id`, `chunk_id`, `source`, `year`, `stage`, `quote_author`, `historical_period`, `interpretation_type` и другие поля. |
| Построить индекс | В `1_ingest.ipynb` создаётся локальный Qdrant index в `data/vectorstore/qdrant`. |
| Проверить, что metadata возвращаются | В `2_search_demo.ipynb` и `3_agent_demo.ipynb` результаты содержат `document_id`, `chunk_id`, `source` и автора. |
| Подключить MCP-сервер | Реализован MCP-сервер `src/hw_rag_mcp/mcp_server.py` на FastMCP. |
| Показать MCP-инструмент | В `3_agent_demo.ipynb` вручную вызываются MCP tools. |
| Подключить MCP tool к агенту | MCP tools подключаются к LangChain agent через `langchain-mcp-adapters`. |
| Вернуть top-k результаты | Tools возвращают top-k результатов с `document_id`, `chunk_id`, `source`, `author`, `content`. |
| Провести проверку на 15–20 запросах | Создан eval-набор из 20 запросов в `data/eval/eval_queries.json`. |
| Дать ручную оценку | Ручной анализ лежит в `notebooks/eval_report.md`. |
| Приложить инструкцию запуска | Этот README описывает установку, env, порядок запуска notebooks, MCP и eval. |

## Что получилось хорошо

- Стабильные `chunk_id` позволяют ссылаться на конкретные темы.
- Metadata не теряются при индексации и поиске.
- Semantic search хорошо работает для широких смысловых запросов.
- Metadata search хорошо работает для авторов цитат.
- Metadata-aware reranking улучшает запросы по типу авторской позиции.
- Агент умеет вызывать MCP tools и возвращать найденные фрагменты.
- Eval показывает, где система работает хорошо, а где retrieval требует улучшения.

## Ограничения

1. MCP-сервер является собственным adapter layer на FastMCP, а не готовым сервером конкретной векторной базы вроде `mcp-server-qdrant`.
2. Агентный ответ не полностью детерминирован: модель иногда добавляет собственные формулировки вокруг найденного `content`.
3. В eval не сохраняется структурированный trace фактического tool call; ручная оценка делается по финальному ответу агента и при необходимости по Langfuse UI.
4. Сложные causal-запросы могут требовать более точной стратегии retrieval или отдельной настройки.

## Полезные команды

Установка зависимостей:

```bash
uv sync
```

Запуск MCP-сервера вручную:

```bash
uv run python -m hw_rag_mcp.mcp_server
```

Печать структуры проекта:

```bash
tree -I '.venv|__pycache__' > tree.txt
```

Если локальный Qdrant заблокирован другим процессом:

```bash
lsof | grep "data/vectorstore/qdrant"
kill <PID>
```

## Рекомендуемый порядок проверки преподавателем

1. Прочитать `README.md`.
2. Посмотреть структуру проекта и `mcp_config.json`.
3. Открыть `notebooks/1_ingest.ipynb` и проверить ingest/indexing pipeline.
4. Открыть `notebooks/2_search_demo.ipynb` и проверить retrieval experiments.
5. Открыть `notebooks/3_agent_demo.ipynb` и проверить MCP + agent demo.
6. Посмотреть `data/eval/eval_queries.json` и `data/eval/eval_results.json`.
7. Прочитать `notebooks/eval_report.md`.

