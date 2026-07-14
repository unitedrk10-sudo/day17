from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

NOTES: list[str] = []
app = FastAPI(title="Notes API")
HTML_PATH = Path(__file__).with_name("notes.html")


@app.get("/")
def home():
    """브라우저에서 노트 UI를 보여줍니다."""
    return FileResponse(HTML_PATH)


def list_notes() -> dict:
    return {"notes": NOTES, "count": len(NOTES)}


def add_note(text: str) -> dict:
    NOTES.append(text)
    return {"ok": True, "notes": list(NOTES)}


def delete_note(index: int) -> dict:
    if index < 0 or index >= len(NOTES):
        return {"ok": False, "error": f"index {index} 없음"}
    removed = NOTES.pop(index)
    return {"ok": True, "removed": removed, "notes": list(NOTES)}


@app.get("/notes")
def api_list_notes():
    return list_notes()


@app.post("/notes")
def api_add_note(text: str):
    return add_note(text)


@app.delete("/notes/{index}")
def api_delete_note(index: int):
    result = delete_note(index)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result