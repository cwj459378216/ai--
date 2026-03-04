import os
from pathlib import Path

try:
	from dotenv import load_dotenv

	root_env = Path(__file__).resolve().parents[1] / ".env"
	local_env = Path(__file__).resolve().parent / ".env"
	load_dotenv(dotenv_path=root_env)
	load_dotenv(dotenv_path=local_env)
except Exception:
	pass


BASE_DIR = Path(__file__).resolve().parents[1]

DOC_DIR = BASE_DIR / "doc"
DOC1_DIR = DOC_DIR / "doc1"
DOC2_DIR = DOC_DIR / "doc2"

VECTOR_DIR = BASE_DIR / "vectors"
MILVUS_DB_PATH = VECTOR_DIR / "milvus.db"

DEFAULT_COLLECTION = "rag_demo"
DEFAULT_TOPK = 3

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
DASHSCOPE_LLM_MODEL = os.getenv("DASHSCOPE_LLM_MODEL", "qwen-plus")
DASHSCOPE_SMALL_LLM_MODEL = os.getenv("DASHSCOPE_SMALL_LLM_MODEL", "qwen-turbo")
LLM_BASE_URL = os.getenv(
	"LLM_BASE_URL",
	"https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")

DEFAULT_LLM_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_default_llm_base = DEFAULT_LLM_BASE
if os.getenv("EMBEDDING_BASE_URL") is not None:
	EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
elif LLM_BASE_URL and LLM_BASE_URL != _default_llm_base:
	EMBEDDING_BASE_URL = LLM_BASE_URL.rstrip("/")
else:
	EMBEDDING_BASE_URL = ""
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

IS_LOCAL_LLM = bool(LLM_BASE_URL and LLM_BASE_URL != DEFAULT_LLM_BASE)
IS_LOCAL_EMBEDDING = bool(EMBEDDING_BASE_URL)

# 控制开关
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "1") == "1"
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "1") == "1"
