from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

PROJECT_DIR = Path(__file__).parent
ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_MAIN_MODEL = "ollama:gemma4:26b"
DEFAULT_LLM_PROVIDER = "ollama"

_KNOWN_PROVIDER_PREFIXES: dict[str, str] = {
    "ollama:": "ollama",
    "llama_cpp:": "llama_cpp",
    "llamacpp:": "llama_cpp",
    "llama_server:": "llama_server",
    "llamaserver:": "llama_server",
    "huggingface:": "huggingface",
    "hf:": "huggingface",
}

# Cache for ChatLlamaCpp instances keyed by (model_path, temperature).
# llama_cpp loads the model in-process; reuse the same object across all agents
# to avoid loading the same large GGUF file multiple times.
_LLAMA_CPP_INSTANCE_CACHE: dict[
    tuple[str, float, int, int, int, int, bool],
    Any,
] = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

AGENT_MODEL_ENV_KEYS: dict[str, str] = {
    "main": "DEEPAGENT_MODEL",
    "planner": "PLANNER_MODEL",
    "websearch": "WEBSEARCH_MODEL",
    "writer": "WRITER_MODEL",
    "coder": "CODER_MODEL",
    "reviewer": "REVIEWER_MODEL",
    "presenter": "PRESENTER_MODEL",
    "data_scientist": "DATA_SCIENTIST_MODEL",
    "ragsub": "RAG_SUB_MODEL",
}

AGENT_MODEL_LABELS: dict[str, str] = {
    "main": "Main Agent",
    "planner": "Planner",
    "websearch": "Web Search",
    "writer": "Writer",
    "coder": "Coder",
    "reviewer": "Reviewer",
    "presenter": "Presenter",
    "data_scientist": "Data Scientist",
    "ragsub": "RAG SubAgent",
}


def get_env_value(key: str, default: str = "") -> str:
    file_values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    file_value = file_values.get(key)
    if file_value is not None and str(file_value).strip() != "":
        return str(file_value).strip()
    return os.getenv(key, default).strip()


def get_main_model() -> str:
    return get_env_value(AGENT_MODEL_ENV_KEYS["main"], DEFAULT_MAIN_MODEL) or DEFAULT_MAIN_MODEL


def get_agent_model_override(agent_name: str) -> str:
    env_key = AGENT_MODEL_ENV_KEYS.get(agent_name)
    if agent_name == "websearch":
        return get_env_value(env_key, "") or get_env_value("RESEARCHER_MODEL", "")
    if not env_key or agent_name == "main":
        return ""
    return get_env_value(env_key, "")


def get_agent_model(agent_name: str) -> str:
    if agent_name == "main":
        return get_main_model()
    return get_agent_model_override(agent_name) or get_main_model()


def set_env_value(key: str, value: str) -> None:
    env_content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    replacement = f"{key}={value}"
    if re.search(rf"(?m)^{re.escape(key)}=", env_content):
        env_content = re.sub(
            rf"(?m)^{re.escape(key)}=.*$",
            replacement,
            env_content,
        )
    else:
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_content += replacement + "\n"
    ENV_FILE.write_text(env_content, encoding="utf-8")
    os.environ[key] = value


def reload_env_file() -> None:
    load_dotenv(dotenv_path=ENV_FILE, override=True)


def get_default_llm_provider() -> str:
    provider = get_env_value("DEEPAGENT_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()
    if provider in ("ollama", "llama_cpp", "llama_server", "huggingface"):
        return provider
    return DEFAULT_LLM_PROVIDER


def resolve_provider_and_model(model_name: str) -> tuple[str, str]:
    raw = (model_name or "").strip()
    if not raw:
        return get_default_llm_provider(), ""

    lower = raw.lower()
    for prefix, provider in _KNOWN_PROVIDER_PREFIXES.items():
        if lower.startswith(prefix):
            return provider, raw[len(prefix):].strip()

    return get_default_llm_provider(), raw


def _build_llama_server_chat_model(model_name: str, temperature: float) -> Any:
    """Connect to a running llama-server via its OpenAI-compatible REST API.

    Start the server with:
        llama-server.exe -m model.gguf -c 8192 -ngl -1 --port 8080

    Set LLAMA_SERVER_BASE_URL to override the default http://localhost:8080/v1.
    The model name sent in requests is ignored by llama-server but required by
    the OpenAI client; any non-empty string works.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "llama_server requires the `langchain-openai` package. "
            "Install it in the active environment and retry."
        ) from exc

    base_url = os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1").strip()
    # llama-server ignores the model field but ChatOpenAI requires it.
    resolved_model = model_name or os.getenv("LLAMA_SERVER_MODEL", "local").strip() or "local"
    max_tokens = int(os.getenv("LLAMA_CPP_MAX_TOKENS", "1024"))
    return ChatOpenAI(
        base_url=base_url,
        api_key="llama-server",  # dummy — server has no auth
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=300,
    )


def _build_ollama_chat_model(model_name: str, temperature: float) -> Any:
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    model = model_name or os.getenv("OLLAMA_MODEL", "gemma4:26b")
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def _build_llama_cpp_chat_model(model_name: str, temperature: float) -> Any:
    from langchain_community.chat_models import ChatLlamaCpp

    model_path = model_name or os.getenv("LLAMA_CPP_MODEL_PATH", "").strip()
    if not model_path:
        raise ValueError(
            "llama.cpp requires a model path. Set LLAMA_CPP_MODEL_PATH or use "
            "DEEPAGENT_MODEL=llama_cpp:/absolute/path/to/model.gguf"
        )

    n_ctx = int(os.getenv("LLAMA_CPP_N_CTX", "8192"))
    n_batch = int(os.getenv("LLAMA_CPP_N_BATCH", "256"))
    max_tokens = int(os.getenv("LLAMA_CPP_MAX_TOKENS", "1024"))
    n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "-1"))
    flash_attn = _env_bool("LLAMA_CPP_FLASH_ATTN", True)

    # Reuse the same ChatLlamaCpp instance only when all runtime-critical
    # settings match, so UI/env changes apply immediately.
    cache_key = (
        model_path,
        temperature,
        n_ctx,
        n_batch,
        max_tokens,
        n_gpu_layers,
        flash_attn,
    )
    if cache_key in _LLAMA_CPP_INSTANCE_CACHE:
        return _LLAMA_CPP_INSTANCE_CACHE[cache_key]

    instance = ChatLlamaCpp(
        model_path=model_path,
        temperature=temperature,
        n_ctx=n_ctx,
        n_batch=n_batch,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers,
        flash_attn=flash_attn,
        verbose=False,
    )
    _LLAMA_CPP_INSTANCE_CACHE[cache_key] = instance
    return instance


def _build_huggingface_chat_model(model_name: str, temperature: float) -> Any:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    repo_id = model_name or os.getenv("HUGGINGFACE_REPO_ID", "").strip()
    if not repo_id:
        raise ValueError(
            "Hugging Face requires a repo id. Set HUGGINGFACE_REPO_ID or use "
            "DEEPAGENT_MODEL=huggingface:<repo-id>"
        )

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token or None,
        task=os.getenv("HUGGINGFACE_TASK", "text-generation"),
        temperature=temperature,
        max_new_tokens=int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "1024")),
    )
    return ChatHuggingFace(llm=endpoint)


def create_chat_model(model_name: str, temperature: float = 0) -> Any:
    """Create a chat model from a provider-prefixed model string or provider env.

    Supported providers:
    - ollama: ollama:model_tag or DEEPAGENT_LLM_PROVIDER=ollama
    - llama_cpp: llama_cpp:/path/to/model.gguf or DEEPAGENT_LLM_PROVIDER=llama_cpp
    - huggingface: huggingface:repo/id or DEEPAGENT_LLM_PROVIDER=huggingface
    """
    provider, resolved_model = resolve_provider_and_model(model_name)

    if provider == "ollama":
        return _build_ollama_chat_model(resolved_model, temperature)
    if provider == "llama_cpp":
        return _build_llama_cpp_chat_model(resolved_model, temperature)
    if provider == "llama_server":
        return _build_llama_server_chat_model(resolved_model, temperature)
    if provider == "huggingface":
        return _build_huggingface_chat_model(resolved_model, temperature)

    raise ValueError(f"Unsupported LLM provider: {provider}")
