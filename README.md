## ai2：模块化 RAG（Milvus-lite + LlamaIndex + DashScope/百炼）

本项目提供一个“可复用的模块化 RAG”实现：

- **向量化**：把 doc/doc1 下的文件分块后写入 Milvus-lite 本地库
- **检索问答**：TopK 召回 +（可选）问题改写 +（可选）轻量重排序 + 生成回答
- **两种入口**：命令行（CLI）与图形界面（Streamlit）

更完整的模块与参数说明见：docs/rag_modular.md

### 快速开始（macOS/Linux）

1) 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate
```

2) 安装依赖

```bash
pip install -r requirements.txt
```

3) 配置环境变量（至少需要 Key）

```bash
export DASHSCOPE_API_KEY="你的 Key"
```

也可以在项目根目录放置 `.env` 文件（`code/config.py` 会自动加载）。

4) 放入待向量化文件

将 `.txt` / `.pdf` / `.docx` / `.csv` 放到 `doc/doc1/`。

5) 向量化入库

```bash
python -m code.vectorize_cli
```

6) CLI 提问

```bash
python -m code.cli -q "你的问题"
```

7) 启动 GUI

```bash
streamlit run code/gui.py
```

### 常用环境变量

- `DASHSCOPE_API_KEY` 或 `BAILIAN_API_KEY`：必填（除非你配置了本地 LLM/Embedding）
- `DASHSCOPE_LLM_MODEL`：生成模型（默认 `qwen-plus`）
- `DASHSCOPE_SMALL_LLM_MODEL`：改写模型（默认 `qwen-turbo`）
- `ENABLE_QUERY_REWRITE=0`：关闭问题改写
- `ENABLE_RERANK=0`：关闭重排序

