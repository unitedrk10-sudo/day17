"""samples PDF 정리: 카탈로그/토큰 유사도 분류 + Agent 도구."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent
SAMPLES_DIR = BASE / "samples"
PDF_SAMPLES_DIR = SAMPLES_DIR / "pdf_samples"
CATALOG_DIR = PDF_SAMPLES_DIR / "_catalog"
WORK_DIR = SAMPLES_DIR / "_pdf_work"
TXT_DIR = WORK_DIR / "txt"
SUMMARY_DIR = WORK_DIR / "summaries"
ORGANIZED_DIR = SAMPLES_DIR / "_organized"
DEFAULT_OUTPUT_DIR = PDF_SAMPLES_DIR / "_organized_by_topic"

SKIP_PARTS = {"_pdf_work", "_organized", "_organized_by_topic", "_catalog"}
ROOT_ONLY_PDFS = {"Language_Models.pdf"}
MIN_OVERLAP = 15

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "pdf", "paper", "conference", "proceedings", "abstract", "introduction",
    "및", "등", "것", "수", "있", "하는", "대한", "위한", "통해", "따라",
    "문서", "요약", "제목", "저자", "내용", "경우", "또는", "에서", "으로",
    "핵심", "주제", "분야", "연구", "논문", "모델", "기술", "방법", "결과",
}

load_dotenv(BASE.parent / ".env")
client = OpenAI()


# ── 공통 ──────────────────────────────────────────


def slugify_topic(name: str) -> str:
    cleaned = re.sub(r"[^\w가-힣\- ]+", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "기타"


def safe_folder_name(name: str, max_len: int = 48) -> str:
    return slugify_topic(name)[:max_len]


def tokenize(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[가-힣a-zA-Z0-9]+", text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    }


def find_pdfs(root: Path = SAMPLES_DIR) -> list[Path]:
    pdfs: list[Path] = []
    for path in sorted(root.rglob("*.pdf")):
        rel_parts = set(path.relative_to(root).parts)
        if rel_parts & SKIP_PARTS:
            continue
        pdfs.append(path)
    return pdfs


def find_pdf_by_name(name: str) -> Path | None:
    for path in find_pdfs():
        if path.name == name:
            return path
    return None


def pdf_to_text(pdf_path: Path, txt_dir: Path) -> Path:
    txt_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    txt_path = txt_dir / f"{pdf_path.stem}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return txt_path


def summarize_text(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "다음 문서를 5~8문장으로 요약하세요. "
                    "주제, 분야, 문서 유형(논문/규정/보도자료 등), 핵심 키워드를 포함하세요."
                ),
            },
            {"role": "user", "content": text[:12_000]},
        ],
    )
    return response.choices[0].message.content or ""


def load_catalog_index() -> list[dict]:
    index_path = CATALOG_DIR / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8")).get("documents", [])


def read_catalog_summary(summary_file: str) -> str:
    path = CATALOG_DIR / summary_file
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ── 붙여넣은 코드 장점: 카탈로그·휴리스틱 ─────────


def list_pdf_files() -> str:
    names = sorted(p.name for p in PDF_SAMPLES_DIR.glob("*.pdf"))
    all_names = sorted(p.name for p in find_pdfs())
    return json.dumps(
        {"pdf_samples": names, "all_samples": all_names},
        ensure_ascii=False,
        indent=2,
    )


def read_catalog_metadata() -> str:
    docs = []
    for doc in load_catalog_index():
        summary_text = read_catalog_summary(doc.get("summary_file", ""))
        one_line = ""
        lines = summary_text.splitlines()
        for i, line in enumerate(lines):
            if "한 줄 요약" in line and i + 1 < len(lines):
                one_line = lines[i + 1].strip()
                break
        docs.append(
            {
                "pdf_name": doc.get("pdf_name", ""),
                "category": doc.get("category", "기타"),
                "keywords": doc.get("keywords", []),
                "one_line_summary": one_line,
            }
        )
    return json.dumps({"documents": docs}, ensure_ascii=False, indent=2)


def _heuristic_topic(category: str, keywords: list[str]) -> str:
    c = (category or "").strip().lower()
    kw = " ".join(k.lower() for k in (keywords or []))

    if "규정" in c or "학칙" in kw:
        return "학교규정"
    if "보도자료" in c or "경영실적" in kw or "sk하이닉스" in kw:
        return "기업보도자료"
    if "논문" in c:
        if any(t in kw for t in ["vision", "vit", "image", "contrastive"]):
            return "논문_비전AI"
        if any(t in kw for t in ["speech", "language", "multimodal", "data2vec", "physics"]):
            return "논문_멀티모달"
        if any(t in kw for t in ["agent", "llm", "autonomous"]):
            return "논문_에이전트"
        return "논문_기타"
    return "기타"


def propose_topic_groups(max_topics: int = 8, min_docs_per_topic: int = 1) -> str:
    raw = json.loads(read_catalog_metadata())
    groups: dict[str, list[str]] = {}
    for doc in raw.get("documents", []):
        topic = _heuristic_topic(doc.get("category", ""), doc.get("keywords", []))
        name = doc.get("pdf_name", "")
        if name and find_pdf_by_name(name):
            groups.setdefault(topic, []).append(name)

    filtered = [
        {"topic": topic, "pdf_files": sorted(files)}
        for topic, files in groups.items()
        if len(files) >= max(1, min_docs_per_topic)
    ]
    filtered.sort(key=lambda x: (-len(x["pdf_files"]), x["topic"]))
    return json.dumps({"groups": filtered[:max_topics]}, ensure_ascii=False, indent=2)


def catalog_topic_for_pdf(pdf_name: str) -> str | None:
    for doc in load_catalog_index():
        if doc.get("pdf_name") == pdf_name:
            return _heuristic_topic(doc.get("category", ""), doc.get("keywords", []))
    return None


# ── 이동/복원 (copy·move·dry_run) ─────────────────


def transfer_pdf(
    src: Path,
    dest_dir: Path,
    mode: str = "move",
    dry_run: bool = False,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        return dest
    if dest.exists() and mode == "move":
        stem, suffix = src.stem, src.suffix
        n = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{n}{suffix}"
            n += 1

    if dry_run:
        return dest

    if mode == "copy":
        shutil.copy2(src, dest)
    else:
        shutil.move(str(src), str(dest))
    return dest


def apply_topic_organization(
    groups_json: str,
    mode: str = "copy",
    dry_run: bool = False,
    output_dir: str | None = None,
) -> str:
    try:
        payload = json.loads(groups_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"groups_json 파싱 실패: {e}"}, ensure_ascii=False)

    groups = payload.get("groups", [])
    out_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    operations, errors = [], []

    for item in groups:
        topic = slugify_topic(item.get("topic", "기타"))
        topic_dir = out_root / topic
        if not dry_run:
            topic_dir.mkdir(parents=True, exist_ok=True)

        for name in item.get("pdf_files", []):
            src = find_pdf_by_name(name)
            if not src:
                errors.append(f"파일 없음: {name}")
                continue
            try:
                dst = transfer_pdf(src, topic_dir, mode=mode, dry_run=dry_run)
                operations.append(
                    {"topic": topic, "action": mode, "src": str(src), "dst": str(dst)}
                )
            except OSError as e:
                errors.append(f"{name}: {e}")

    manifest = {
        "strategy": "catalog",
        "output_dir": str(out_root),
        "mode": mode,
        "dry_run": dry_run,
        "total_operations": len(operations),
        "errors": errors,
        "operations": operations,
    }
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / "_organization_manifest.json"
        log_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(log_path)
    return json.dumps(manifest, ensure_ascii=False, indent=2)


def infer_original_path(pdf_name: str, organized_filename: str) -> Path:
    if pdf_name in ROOT_ONLY_PDFS:
        return SAMPLES_DIR / pdf_name
    if pdf_name == "학칙.pdf" and organized_filename != "학칙.pdf":
        return PDF_SAMPLES_DIR / pdf_name
    if pdf_name == "학칙.pdf":
        return SAMPLES_DIR / pdf_name
    return PDF_SAMPLES_DIR / pdf_name


def restore_organized(
    samples_dir: Path = SAMPLES_DIR,
    organized_dir: Path = ORGANIZED_DIR,
    manifest_path: Path | None = None,
) -> list[dict]:
    manifest_path = manifest_path or (WORK_DIR / "organize_manifest.json")
    restored: list[dict] = []

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in data.get("moved", []):
            src = samples_dir / item["dest"].replace("\\", "/")
            if not src.exists():
                continue
            if "src" in item:
                dest = samples_dir / item["src"].replace("\\", "/")
            else:
                dest = infer_original_path(item["pdf"], src.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            print(f"  복원: {src.name} → {dest.relative_to(samples_dir)}")
            restored.append({"from": str(src), "to": str(dest)})
    else:
        for pdf_path in sorted(organized_dir.rglob("*.pdf")):
            dest = infer_original_path(pdf_path.name, pdf_path.name)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pdf_path), str(dest))
            restored.append({"from": str(pdf_path), "to": str(dest)})

    print(f"복원 완료: {len(restored)}개")
    return restored


# ── 토큰 유사도 분류 (기존 방식) ─────────────────


def summarize_pdf(
    pdf_path: Path,
    txt_dir: Path,
    summary_dir: Path,
    reuse_cached: bool = True,
) -> tuple[Path, Path, set[str]]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{pdf_path.stem}.summary.txt"
    txt_path = txt_dir / f"{pdf_path.stem}.txt"

    catalog_doc = next((d for d in load_catalog_index() if d.get("pdf_name") == pdf_path.name), None)
    if catalog_doc:
        cat_summary = read_catalog_summary(catalog_doc.get("summary_file", ""))
        if cat_summary:
            summary_path.write_text(cat_summary, encoding="utf-8")
            if not txt_path.exists():
                pdf_to_text(pdf_path, txt_dir)
            return txt_path, summary_path, tokenize(cat_summary)

    if reuse_cached and summary_path.exists():
        if not txt_path.exists():
            pdf_to_text(pdf_path, txt_dir)
        summary = summary_path.read_text(encoding="utf-8")
        return txt_path, summary_path, tokenize(summary)

    if not txt_path.exists():
        pdf_to_text(pdf_path, txt_dir)
    summary = summarize_text(txt_path.read_text(encoding="utf-8"))
    summary_path.write_text(summary, encoding="utf-8")
    return txt_path, summary_path, tokenize(summary)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            grouped.setdefault(self.find(i), []).append(i)
        return grouped


def cluster_by_overlap(token_sets: list[set[str]], min_overlap: int) -> list[list[int]]:
    uf = UnionFind(len(token_sets))
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            if len(token_sets[i] & token_sets[j]) >= min_overlap:
                uf.union(i, j)
    return list(uf.groups().values())


def folder_name_for_cluster(
    members: list[int],
    records: list[dict],
    token_sets: list[set[str]],
) -> str:
    topics = [catalog_topic_for_pdf(records[i]["pdf_path"].name) for i in members]
    topics = [t for t in topics if t]
    if topics and len(set(topics)) == 1:
        return slugify_topic(topics[0])

    matched: set[str] = set()
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            matched |= token_sets[a] & token_sets[b]
    freq: Counter[str] = Counter()
    for idx in members:
        for token in token_sets[idx]:
            if token in matched:
                freq[token] += 1
    if freq:
        return slugify_topic(freq.most_common(1)[0][0])
    return slugify_topic(records[members[0]]["pdf_path"].stem)


def organize_by_catalog(
    mode: str = "move",
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """카탈로그 category/keywords 휴리스틱으로 분류 (API 불필요)."""
    groups_json = propose_topic_groups()
    groups = json.loads(groups_json).get("groups", [])
    unknown = [
        p.name
        for p in find_pdfs()
        if catalog_topic_for_pdf(p.name) is None
    ]
    if unknown:
        groups.append({"topic": "기타", "pdf_files": sorted(unknown)})

    result = json.loads(
        apply_topic_organization(
            json.dumps({"groups": groups}, ensure_ascii=False),
            mode=mode,
            dry_run=dry_run,
            output_dir=str(output_dir or ORGANIZED_DIR),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def organize_by_tokens(
    samples_dir: Path = SAMPLES_DIR,
    min_overlap: int = MIN_OVERLAP,
    mode: str = "move",
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """PDF→txt→요약→토큰 유사도 분류."""
    out_root = output_dir or ORGANIZED_DIR
    pdfs = find_pdfs(samples_dir)
    if not pdfs:
        print("정리할 PDF가 없습니다.")
        return {}

    TXT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for pdf_path in pdfs:
        print(f"  처리: {pdf_path.name}")
        _, summary_path, tokens = summarize_pdf(pdf_path, TXT_DIR, SUMMARY_DIR)
        records.append({"pdf_path": pdf_path, "tokens": tokens})

    token_sets = [r["tokens"] for r in records]
    clusters = cluster_by_overlap(token_sets, min_overlap)
    moved: list[dict] = []

    print("\n분류:")
    for members in clusters:
        if len(members) == 1:
            idx = members[0]
            folder = slugify_topic(records[idx]["pdf_path"].stem)
        else:
            folder = folder_name_for_cluster(members, records, token_sets)

        dest_dir = out_root / folder
        print(f"  [{folder}] ({len(members)}개)")
        for idx in members:
            src = records[idx]["pdf_path"]
            dest = transfer_pdf(src, dest_dir, mode=mode, dry_run=dry_run)
            print(f"    → {dest.relative_to(samples_dir)}")
            moved.append(
                {
                    "pdf": src.name,
                    "src": str(src.relative_to(samples_dir)),
                    "folder": folder,
                    "dest": str(dest.relative_to(samples_dir)),
                }
            )

    manifest = {"strategy": "token", "min_overlap": min_overlap, "mode": mode, "moved": moved}
    if not dry_run:
        manifest_path = WORK_DIR / "organize_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def organize_hybrid(
    min_overlap: int = MIN_OVERLAP,
    mode: str = "move",
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """카탈로그 문서는 휴리스틱, 나머지는 토큰 유사도."""
    out_root = output_dir or ORGANIZED_DIR
    catalog_names = {d.get("pdf_name") for d in load_catalog_index()}
    catalog_pdfs = [p for p in find_pdfs() if p.name in catalog_names]
    other_pdfs = [p for p in find_pdfs() if p.name not in catalog_names]

    results = {"catalog": None, "token": None}

    if catalog_pdfs:
        groups: dict[str, list[str]] = {}
        for path in catalog_pdfs:
            topic = catalog_topic_for_pdf(path.name) or "기타"
            groups.setdefault(topic, []).append(path.name)
        payload = {"groups": [{"topic": t, "pdf_files": f} for t, f in groups.items()]}
        results["catalog"] = json.loads(
            apply_topic_organization(
                json.dumps(payload, ensure_ascii=False),
                mode=mode,
                dry_run=dry_run,
                output_dir=str(out_root),
            )
        )

    if other_pdfs:
        records = []
        TXT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        for pdf_path in other_pdfs:
            _, _, tokens = summarize_pdf(pdf_path, TXT_DIR, SUMMARY_DIR)
            records.append({"pdf_path": pdf_path, "tokens": tokens})
        token_sets = [r["tokens"] for r in records]
        for members in cluster_by_overlap(token_sets, min_overlap):
            folder = (
                slugify_topic(records[members[0]]["pdf_path"].stem)
                if len(members) == 1
                else folder_name_for_cluster(members, records, token_sets)
            )
            for idx in members:
                transfer_pdf(records[idx]["pdf_path"], out_root / folder, mode=mode, dry_run=dry_run)
        results["token"] = {"count": len(other_pdfs), "min_overlap": min_overlap}

    return results


# ── Agent (Tool Calling) ─────────────────────────


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_pdf_files",
            "description": "samples/pdf_samples 및 전체 samples PDF 목록.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_catalog_metadata",
            "description": "_catalog/index.json과 summary 메타데이터.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_topic_groups",
            "description": "카탈로그 기반 주제별 그룹 제안.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_topics": {"type": "integer"},
                    "min_docs_per_topic": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_topic_organization",
            "description": "주제별 폴더 생성 후 PDF copy/move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "groups_json": {"type": "string"},
                    "mode": {"type": "string", "enum": ["copy", "move"]},
                    "dry_run": {"type": "boolean"},
                    "output_dir": {"type": "string"},
                },
                "required": ["groups_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_by_tokens_tool",
            "description": "요약+토큰 유사도로 자동 분류 (min_overlap 조절 가능).",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_overlap": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["copy", "move"]},
                    "dry_run": {"type": "boolean"},
                },
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_pdf_files": lambda **_: list_pdf_files(),
    "read_catalog_metadata": lambda **_: read_catalog_metadata(),
    "propose_topic_groups": propose_topic_groups,
    "apply_topic_organization": apply_topic_organization,
    "organize_by_tokens_tool": lambda **kw: json.dumps(
        organize_by_tokens(
            min_overlap=kw.get("min_overlap", MIN_OVERLAP),
            mode=kw.get("mode", "move"),
            dry_run=kw.get("dry_run", False),
        ),
        ensure_ascii=False,
        indent=2,
    ),
}

AGENT_SYSTEM = """
너는 pdf_samples 문서 정리 Agent다.

도구 선택:
- 목록/메타 확인 → list_pdf_files, read_catalog_metadata
- 카탈로그 기반 주제 분류 → propose_topic_groups → apply_topic_organization
- 요약 토큰 유사도 분류 → organize_by_tokens_tool

규칙:
- 사용자가 따로 말하지 않으면 mode='copy', dry_run=false
- 카탈로그에 있는 PDF는 propose_topic_groups 우선
- 최종 답변: 폴더 경로, 처리 개수, 오류 유무 (한국어)
""".strip()


def run_agent(command: str, max_rounds: int = 8) -> str:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": command},
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
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "tool 호출 최대 횟수 초과"


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "hybrid"

    if cmd == "restore":
        restore_organized()
    elif cmd == "catalog":
        organize_by_catalog(mode="move")
    elif cmd == "token":
        organize_by_tokens(min_overlap=MIN_OVERLAP, mode="move")
    elif cmd == "agent":
        print(
            run_agent(
                "pdf_samples PDF를 주제별로 폴더 생성해서 정리해줘. "
                "카탈로그 기반으로 먼저 시도하고, 기본은 copy로."
            )
        )
    elif cmd == "dry":
        organize_by_catalog(mode="copy", dry_run=True)
    else:
        print("=== hybrid: 카탈로그 + 토큰 ===")
        organize_hybrid(min_overlap=MIN_OVERLAP, mode="move")
