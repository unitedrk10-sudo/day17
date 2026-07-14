from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

# ── 경로 ──────────────────────────────────────────
BASE = Path(__file__).resolve().parent
load_dotenv(BASE.parent / ".env")
DOC_LIBRARY = BASE / "samples" / "pdf_samples"
CATALOG_DIR = DOC_LIBRARY / "_catalog"

client = OpenAI()
INDEX_CACHE: dict[str, list[dict]] = {}


def split_text(text: str, chunk_size: int = 280, overlap: int = 60) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[가-힣a-zA-Z0-9]+", text) if len(t) > 1}


def search_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    q = tokenize(query)
    scored = []
    for item in chunks:
        score = len(q & tokenize(item["text"]))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**it, "score": s} for s, it in scored[:top_k]]


def read_pdf_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def chunk_stem(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    return re.sub(r"[^\w가-힣\-]+", "_", stem)[:40].strip("_")


def build_pdf_index(pdf_name: str) -> list[dict]:
    if pdf_name in INDEX_CACHE:
        return INDEX_CACHE[pdf_name]
    pdf_path = DOC_LIBRARY / pdf_name
    if not pdf_path.exists():
        return []
    text = read_pdf_text(pdf_path)
    prefix = chunk_stem(pdf_name)
    index = [
        {
            "chunk_id": f"{prefix}_C{i:03d}",
            "source_pdf": pdf_name,
            "text": chunk,
        }
        for i, chunk in enumerate(split_text(text), 1)
    ]
    INDEX_CACHE[pdf_name] = index
    return index


def list_documents() -> str:
    """pdf_samples 안 PDF 파일명 목록."""
    names = sorted(p.name for p in DOC_LIBRARY.glob("*.pdf"))
    return json.dumps({"count": len(names), "pdf_files": names}, ensure_ascii=False)


def get_document_catalog() -> str:
    """주어진 _catalog/index.json + summary txt 를 읽어 반환 (생성 X)."""
    index_path = CATALOG_DIR / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = []
    for doc in data["documents"]:
        summary_path = CATALOG_DIR / doc["summary_file"]
        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        one_line = ""
        for i, line in enumerate(summary_text.splitlines()):
            if "한 줄 요약" in line and i + 1 < len(summary_text.splitlines()):
                one_line = summary_text.splitlines()[i + 1].strip()
                break
        entries.append({
            "pdf_name": doc["pdf_name"],
            "category": doc.get("category", ""),
            "keywords": doc.get("keywords", []),
            "one_line_summary": one_line,
        })
    return json.dumps({"documents": entries}, ensure_ascii=False, indent=2)


def search_in_document(pdf_name: str, query: str, top_k: int = 3) -> str:
    """지정 PDF 안에서 RAG 검색."""
    hits = search_chunks(query, build_pdf_index(pdf_name), top_k=top_k)
    return json.dumps(
        {
            "pdf_name": pdf_name,
            "query": query,
            "chunks": hits,
            "message": "검색 결과 없음" if not hits else "ok",
        },
        ensure_ascii=False,
    )



AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "pdf_samples 폴더의 PDF 파일명 목록.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_catalog",
            "description": "문서함 카탈로그(유형·한줄요약·키워드). 어떤 PDF를 열지 정할 때 먼저 호출.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_document",
            "description": "카탈로그로 pdf_name을 고른 뒤, 해당 PDF 내에서 질문과 유사한 조항 검색.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_name": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["pdf_name", "query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_documents": lambda **_: list_documents(),
    "get_document_catalog": lambda **_: get_document_catalog(),
    "search_in_document": search_in_document,
}

AGENT_SYSTEM = """
너는 pdf_samples 문서함 Q&A 도우미입니다.

순서:
1. get_document_catalog 로 문서 목록·한줄요약 확인
2. 질문에 맞는 pdf_name 고른 뒤 search_in_document 호출
3. 도구 결과에 없으면 추측하지 말 것

답변: 한국어 bullet 3개 이내, 마지막 bullet 근거: [pdf_name] chunk_id
""".strip()


def run_agent(question: str, max_rounds: int = 8) -> str:
    """tool_calls 가 없을 때까지 반복."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=messages,
            tools=AGENT_TOOLS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = TOOL_FUNCTIONS[fn](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return "tool 호출 최대 횟수 초과"


DEMO = [
    "문서함에 어떤 PDF가 있어?",
    "석사학위과정 수업연한은?",
    "LLM autonomous agent 서베이 논문의 주제는?",
    "data2vec가 다루는 modality 세 가지는?",
    "2026년 1분기 SK하이닉스 영업이익은?",
    "Vision Transformer embedding이 무슨 특징이 있지?",
]

if __name__ == "__main__":
    for q in DEMO:
        print("=" * 60)
        print("Q:", q)
        print(run_agent(q))
        print()
