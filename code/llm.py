from typing import Any

import requests

from .config import LLM_BASE_URL


def chat_complete(*, api_key: str | None, model: str, messages: list[dict[str, str]]) -> str:
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    resp = requests.post(url, json=data, headers=headers, timeout=120)
    if resp.status_code >= 400:
        body_preview = (resp.text or "").strip()[:1500]
        raise RuntimeError(
            f"LLM API error status={resp.status_code}: {body_preview}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        body_preview = (resp.text or "").strip()[:800]
        raise RuntimeError(
            f"LLM API returned non-JSON response (status={resp.status_code}): {body_preview}"
        ) from exc

    try:
        return payload["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected LLM response shape: keys={list(payload.keys())} "
            f"payload_preview={str(payload)[:800]}"
        ) from exc


def generate_answer(*, api_key: str | None, prompt: str, model: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是公司内部知识数据库管理员和技术文档分析专家。\n\n"
                "你的知识来源仅限于公司内部数据库中的资料，\n"
                "包括但不限于：\n"
                "- 项目手册\n"
                "- 测试文档\n"
                "- 产品说明\n"
                "- 技术方案\n"
                "- PDF 文件\n"
                "- CSV 数据文件\n"
                "- 运维记录\n"
                "- 内部规范文档\n\n"
                "你的职责：\n"
                "1. 基于检索到的资料内容进行分析和回答\n"
                "2. 对文档进行结构化整理\n"
                "3. 提供严谨、客观、技术化的解释\n"
                "4. 不得编造数据库中不存在的信息\n"
                "5. 如果资料不足以支持回答，必须明确说明“资料中未提供相关信息”\n\n"
                "回答要求：\n"
                "- 使用正式、专业、简洁的表达\n"
                "- 优先引用资料中的关键内容\n"
                "- 必要时进行条理化总结（分点说明）\n"
                "- 不进行无依据推测\n"
                "- 不输出与资料无关的内容\n\n"
                "你是内部知识系统的一部分，不是通用聊天助手。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return chat_complete(api_key=api_key, model=model, messages=messages)


def rewrite_question(*, api_key: str | None, question: str, model: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是检索问题改写助手。将用户问题改写为简洁、信息密度高的检索查询，避免冗余。",
        },
        {
            "role": "user",
            "content": f"原问题：{question}\n改写：",
        },
    ]
    return chat_complete(api_key=api_key, model=model, messages=messages)
