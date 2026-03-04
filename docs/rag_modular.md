# RAG 模块化使用说明

这份文档描述本仓库的“模块化 RAG”实现：

- 本地向量库：Milvus-lite（SQLite 文件形式，便于本地开发/演示）
- 索引与检索：LlamaIndex
- 模型侧：DashScope/百炼（默认），也支持 OpenAI 兼容的本地/私有化接口

核心目标是：**把向量化、检索、重排序、生成**拆成可复用模块，并提供 CLI/GUI 两套入口。

## 快速开始

1) 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) 配置 Key

```bash
export DASHSCOPE_API_KEY="你的 Key"   # 或设置 BAILIAN_API_KEY
```

也可用 `.env` 文件（`code/config.py` 会自动加载）：

- 项目根目录：`.env`
- 或模块目录：`code/.env`

3) 放入文档并向量化

```bash
python -m code.vectorize_cli
```

4) 提问（CLI）

```bash
python -m code.cli -q "你的问题"
```

5) 启动 GUI（Streamlit）

```bash
streamlit run code/gui.py
```

提示：如果你当前工作目录在 `code/` 下，也可以运行 `streamlit run gui.py`。

## 目录结构

- `doc/`
  - `doc1/`：未向量化的原始文件（向量化输入目录）
  - `doc2/`：已向量化文件的归档区（可选）
- `vectors/`：向量库存放目录
  - `milvus.db`：Milvus-lite 本地数据库文件（默认）
- `code/`：模块化实现（可复用）
  - `config.py`：路径与环境变量配置（含 `.env` 加载）
  - `loader.py`：文档加载（txt/pdf/docx/csv）与分块
  - `chunking.py`：按字符数分块（带 overlap，优先自然边界）
  - `embedding.py`：Embedding（DashScope/百炼 + OpenAI 兼容）
  - `llm.py`：生成与问题改写（OpenAI 兼容 Chat Completions）
  - `vector_store.py`：Milvus-lite 连接、collection 维度校验、索引构建
  - `vectorize.py`：向量化流程（入库、维度探测、DB 重置、doc1→doc2 归档）
  - `retrieval.py`：检索、上下文格式化、候选合并
  - `rerank.py`：轻量重排序（复用 embedding 做点积打分）
  - `pipeline.py`：RAG 流水线（改写 → 召回 → 重排 → 组 prompt → 生成）
  - `cli.py`：问答 CLI 入口
  - `vectorize_cli.py`：向量化 CLI 入口
  - `clear_cli.py`：清空向量库/重置 doc 目录
  - `gui.py`：图形化界面（Streamlit）
  - `auto_vectorize.py`：自动清理 DB/停止旧 GUI 后再执行向量化（可选工具）

## 支持的文件类型

默认支持：`.txt`、`.pdf`、`.docx`、`.csv`。

- 其他后缀会被跳过，并在控制台提示“未支持的文件类型”。
- PDF 文本提取依赖 `pdfplumber` 的 `extract_text()`，扫描版/图片型 PDF 可能提取为空。

## 配置与环境变量

### 必要配置

- `DASHSCOPE_API_KEY` 或 `BAILIAN_API_KEY`
  - 默认情况下必填
  - 如果你把 `LLM_BASE_URL` 指向本地/私有化 OpenAI 兼容接口，则允许不设置 Key（视你的服务端鉴权而定）

### 常用变量一览

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` / `BAILIAN_API_KEY` | 无 | DashScope/百炼 API Key |
| `DASHSCOPE_LLM_MODEL` | `qwen-plus` | 生成模型 |
| `DASHSCOPE_SMALL_LLM_MODEL` | `qwen-turbo` | 改写模型 |
| `LLM_BASE_URL` | DashScope OpenAI 兼容地址 | Chat Completions Base URL |
| `EMBEDDING_BASE_URL` | 空（跟随逻辑见下） | Embedding 的 OpenAI 兼容 Base URL |
| `EMBEDDING_MODEL` | `text-embedding-v2` | Embedding 模型名 |
| `EMBEDDING_API_KEY` | 空 | Embedding 接口的 Key（仅 OpenAI 兼容模式使用） |
| `EMBEDDING_DIM` | `1536` | 向量维度（会被运行时探测结果覆盖） |
| `ENABLE_QUERY_REWRITE` | `1` | 是否启用问题改写 |
| `ENABLE_RERANK` | `1` | 是否启用重排序 |

补充说明：

- `LLM_BASE_URL` 默认为 DashScope 的 OpenAI 兼容地址：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- `EMBEDDING_BASE_URL` 的生效逻辑：
  - 如果显式设置了 `EMBEDDING_BASE_URL`，则 embedding 一定走 OpenAI 兼容接口
  - 否则，如果 `LLM_BASE_URL` 被改成了非默认值，则 embedding 会自动跟随使用同一个 base URL
  - 否则，embedding 走 DashScope 原生 embedding REST API
- 如果你的 embedding 服务是 Ollama 风格（base URL 形如 `http://localhost:11434/api`），代码会自动用 `prompt` 字段调用。

## 向量库存放位置与重置逻辑

- 默认 DB 路径：`vectors/milvus.db`
- 常见的锁文件：`vectors/.milvus.db.lock`

向量化过程有两类“自动修复”逻辑：

1) **运行时探测维度**：会抽取一段文本调用 embedding，得到真实维度 `dim`，并用它建库/建 collection。
2) **维度不一致自动重置**：当发现已有 collection 的向量维度与当前 embedding 维度不一致时：
   - 默认会重置本地 DB 并重新建库（CLI 可用 `--no-auto-fix-dim` 关闭）

## 命令行用法

### 1) 向量化

```bash
python -m code.vectorize_cli
```

常用参数：

- `--input-dir`：待向量化目录（默认 `doc/doc1`）
- `--collection`：collection 名（默认 `rag_demo`）
- `--chunk-size`：分块大小（默认 500 字符）
- `--overlap`：分块重叠（默认 50 字符）
- `--no-move`：不把 doc1 文件移动到 doc2
- `--no-auto-fix-dim`：禁止“维度不一致自动重置”

示例：

```bash
python -m code.vectorize_cli --chunk-size 600 --overlap 80
python -m code.vectorize_cli --collection rag_demo --no-move
```

### 2) 清空向量库（并重置 doc 目录）

```bash
python -m code.clear_cli
```

说明：

- 会 drop collection（若存在）
- 会删除本地 DB 文件
- 会把 `doc/doc2` 下的文件搬回 `doc/doc1`

### 3) 提问（RAG 问答）

```bash
python -m code.cli -q "你的问题"
```

常用参数：

- `--topk`：召回条数（默认 3）
- `--collection`：collection 名（默认 `rag_demo`）

## GUI 用法（Streamlit）

```bash
streamlit run code/gui.py
```

GUI 提供：

- doc1/doc2 文件列表
- 一键向量化（可调 chunk-size/overlap/是否归档）
- 一键清空（含 DB + doc 目录重置，带确认开关）
- 问答：支持切换 topk、collection、是否改写/重排、选择生成/改写模型

## RAG 流程说明（实现要点）

1) **分块**（见 `chunking.py`）
   - 以字符数为主，带 overlap
   - 优先在自然边界（空行、换行、中文句号等）附近切分，减少截断

2) **检索**（见 `pipeline.py`）
   - 会对“原问题”和“改写问题”（如果开启）分别召回，然后去重合并
   - 召回候选数为 `max(topk * 2, topk)`，最后截断到 topk

3) **重排序**（见 `rerank.py`）
   - 复用同一个 embedding 模型：对每个候选 chunk 再算一次 embedding，并与 query 向量做点积打分
   - 优点：不引入额外 rerank 模型
   - 缺点：会增加请求次数与耗时（候选越多越慢）

4) **生成**（见 `llm.py`）
   - 采用 OpenAI 兼容的 `/chat/completions`
   - 无召回时会 fallback：不带上下文直接回答，但会提示“资料不足”的要求

## 常见问题（Troubleshooting）

### 1) 未召回到任何内容

- 确认已完成向量化（并且向量化写入的 `collection` 与提问时一致）
- 检查 `EMBEDDING_DIM` 是否一致（建议直接清空重建）
- 如果你改过 embedding 服务/模型，建议先执行：

```bash
python -m code.clear_cli
```

### 2) 报错包含 "readonly database" 或 "opened by another program"

- 通常是本地 DB 文件不可写或被占用
- 先关闭正在运行的 Streamlit，再清空重置：

```bash
python -m code.clear_cli
```

也可以使用 `python -m code.auto_vectorize`（会先尝试停止旧的 Streamlit 再向量化）。

### 3) 向量化后 doc1 文件不见了

- 默认会把 `doc/doc1` 下的文件移动到 `doc/doc2` 归档
- 如需保留在 doc1，使用 `--no-move`

### 4) 模型不可用/无权限

- 检查 `DASHSCOPE_LLM_MODEL` / `DASHSCOPE_SMALL_LLM_MODEL` 是否有权限
- 可尝试：`qwen-plus` / `qwen-max` / `qwen-turbo`
