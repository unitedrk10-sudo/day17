"""
측정 오버레이 유무 검출 · all / some / none 분기 Tool.

기획서 ⑧ detect_overlay
------------------------
이미지마다 측정 주석(녹색 측정선 · 컬러 라벨 박스 등)이 있는지 보고,
배치 단위로 다음 경로를 고른다.

- ``all``  : 전부 있음  → ocr_existing
- ``some`` : 일부만   → 있는 장 OCR + 없는 장 learn/measure
- ``none`` : 전부 없음 → edge_measure

하단 스케일바/정보바 밴드는 제외한다 (scale_bar Tool과 역할 분리).

Usage
-----
    from tools.overlay import classify_overlay_batch, route_overlay_mode

    branch = classify_overlay_batch(image_paths)
    next_node = route_overlay_mode(branch)  # 'ocr_existing' | 'learn_and_measure' | 'edge_measure'
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

# sem_edge_tool 과 동일 기준 (하단 정보바·스케일바 제외)
SCALEBAR_Y0 = 0.88
OVERLAY_SCORE_THR = 0.0008

OverlayMode = Literal['all', 'some', 'none']
RouteName = Literal['ocr_existing', 'learn_and_measure', 'edge_measure']


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------

class OverlayImageResult(BaseModel):
    """이미지 1장 오버레이 검출 결과."""

    image_path: str
    has_overlay: bool
    score: float = Field(description='주석 픽셀 / 전체 픽셀 (하단 밴드 제외 후)')
    ok: bool = True
    error: str = ''


class OverlayBranchResult(BaseModel):
    """배치 집계 → LangGraph 분기용."""

    mode: OverlayMode
    results: list[OverlayImageResult]
    with_overlay: list[str] = Field(default_factory=list)
    without_overlay: list[str] = Field(default_factory=list)
    n_total: int = 0
    n_with: int = 0
    n_without: int = 0
    n_failed: int = 0
    score_threshold: float = OVERLAY_SCORE_THR

    @property
    def route(self) -> RouteName:
        return route_overlay_mode(self)


# ---------------------------------------------------------------------------
# I/O (Windows 한글 경로)
# ---------------------------------------------------------------------------

def _load_bgr(image_path: str | Path) -> np.ndarray | None:
    path = Path(image_path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# 단일 검출
# ---------------------------------------------------------------------------

def detect_overlay_one(
    image_path: str | Path,
    *,
    score_threshold: float = OVERLAY_SCORE_THR,
    exclude_bottom: float = SCALEBAR_Y0,
) -> OverlayImageResult:
    """
    한 장에서 (스케일바 제외) 측정 오버레이 유무.

    휴리스틱: HSV에서 녹색 측정선 · 청록/파랑 라벨 · 자홍 각도 호를 집계.
    """
    path = Path(image_path)
    img = _load_bgr(path)
    if img is None:
        return OverlayImageResult(
            image_path=str(path),
            has_overlay=False,
            score=0.0,
            ok=False,
            error='이미지 로드 실패',
        )

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = [
        cv2.inRange(hsv, (35, 70, 70), (95, 255, 255)),    # green lines
        cv2.inRange(hsv, (85, 70, 70), (135, 255, 255)),   # cyan-blue labels
        cv2.inRange(hsv, (140, 50, 70), (175, 255, 255)),  # magenta/pink arc
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    y_cut = int(h * exclude_bottom)
    combined[y_cut:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    score = float((combined > 0).sum()) / float(max(h * w, 1))
    return OverlayImageResult(
        image_path=str(path.resolve()),
        has_overlay=score >= score_threshold,
        score=score,
        ok=True,
    )


# 하위 호환: sem_edge_tool.detect_has_measurement_overlay 와 같은 (bool, score)
def detect_has_measurement_overlay(
    image_path: str | Path,
    *,
    score_threshold: float = OVERLAY_SCORE_THR,
    exclude_bottom: float = SCALEBAR_Y0,
) -> tuple[bool, float]:
    r = detect_overlay_one(
        image_path,
        score_threshold=score_threshold,
        exclude_bottom=exclude_bottom,
    )
    if not r.ok:
        return False, 0.0
    return r.has_overlay, r.score


# ---------------------------------------------------------------------------
# 배치 분기
# ---------------------------------------------------------------------------

def classify_overlay_batch(
    image_paths: list[str | Path],
    *,
    score_threshold: float = OVERLAY_SCORE_THR,
    exclude_bottom: float = SCALEBAR_Y0,
) -> OverlayBranchResult:
    """
    여러 장 검사 후 mode = all | some | none.

    로드 실패 이미지는 without 쪽에 넣고 n_failed에 센다
    (측정 불가로 보고 edge/재시도 경로로 보낼 수 있게).
    """
    if not image_paths:
        return OverlayBranchResult(
            mode='none',
            results=[],
            n_total=0,
            score_threshold=score_threshold,
        )

    results: list[OverlayImageResult] = []
    with_paths: list[str] = []
    without_paths: list[str] = []
    n_failed = 0

    for p in image_paths:
        r = detect_overlay_one(
            p,
            score_threshold=score_threshold,
            exclude_bottom=exclude_bottom,
        )
        results.append(r)
        if not r.ok:
            n_failed += 1
            without_paths.append(r.image_path)
            continue
        if r.has_overlay:
            with_paths.append(r.image_path)
        else:
            without_paths.append(r.image_path)

    n_with = len(with_paths)
    n_without = len(without_paths)
    n_total = len(results)

    if n_with == 0:
        mode: OverlayMode = 'none'
    elif n_without == 0:
        mode = 'all'
    else:
        mode = 'some'

    return OverlayBranchResult(
        mode=mode,
        results=results,
        with_overlay=with_paths,
        without_overlay=without_paths,
        n_total=n_total,
        n_with=n_with,
        n_without=n_without,
        n_failed=n_failed,
        score_threshold=score_threshold,
    )


def route_overlay_mode(
    branch: OverlayBranchResult | OverlayMode | str,
) -> RouteName:
    """
    LangGraph conditional edge 헬퍼.

    all  → ocr_existing
    some → learn_and_measure
    none → edge_measure
    """
    if isinstance(branch, OverlayBranchResult):
        mode = branch.mode
    else:
        mode = str(branch)

    if mode == 'all':
        return 'ocr_existing'
    if mode == 'some':
        return 'learn_and_measure'
    return 'edge_measure'


def summarize_overlay_branch(branch: OverlayBranchResult) -> str:
    """로그/State.summary 용 한 줄."""
    return (
        f'overlay_mode={branch.mode} '
        f'(with={branch.n_with}, without={branch.n_without}, '
        f'failed={branch.n_failed}, route={branch.route})'
    )


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    metro = root / 'data' / 'metrology'
    samples = [
        metro / 'tem_sample_01.png',
        metro / 'synth_scalebar.png',
        metro / 'uploads' / 'synth_repeat_4.png',
        metro / 'uploads' / 'ref_pat.png',
    ]
    paths = [p for p in samples if p.exists()]
    branch = classify_overlay_batch(paths)
    print(summarize_overlay_branch(branch))
    for r in branch.results:
        print(
            f"  [{'O' if r.has_overlay else '-'}] "
            f'{Path(r.image_path).name}  score={r.score:.5f}  ok={r.ok}'
        )
    print('with   :', [Path(p).name for p in branch.with_overlay])
    print('without:', [Path(p).name for p in branch.without_overlay])
