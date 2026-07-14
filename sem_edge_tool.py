"""
SEM/TEM 단면 이미지 edge 측정 툴.

오버레이(숫자 라벨)가 없는 일반 SEM에서 구조물 마스크·좌우 edge를 잡아
height / top·middle·bottom width / sidewall angle 을 측정합니다.

한 이미지에 동일 패턴이 반복되어 있으면 인스턴스별로 각각 측정합니다
(region 예: p0_height, p1_top … / 단일 구조면 height, top … 유지).

첫 장 측정으로 MeasureRecipe를 만들고, 이후 이미지에 recipe를 넘기면
같은 위치·개수로 맞춰 측정합니다. 여러 장은 measure_sem_batch() 사용.

노트북 LangGraph는 detect → overlay|edge 조건부 분기로
폴더/zip 이미지를 처리합니다 (스케일바 제외).

노트북: AIagent_practice.ipynb (§5) 에서 import 하여 사용.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator


class Measurement(BaseModel):
    value: float = Field(description='측정 숫자')
    unit: str = Field(description='단위 (nm, px, ° 등)')
    anchor_x: float = Field(description='측정 앵커 X (0~1)')
    anchor_y: float = Field(description='측정 앵커 Y (0~1)')
    orientation: Literal['horizontal', 'vertical', 'angle'] = Field(
        description='가로/세로/각도 측정'
    )
    region: str = Field(
        default='',
        description='구역: height | top | middle | bottom | right_sidewall 등 '
        '(다중 패턴이면 p0_height, p1_top …)',
    )

    @field_validator('anchor_x', 'anchor_y', mode='after')
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class PatternRef(BaseModel):
    """첫 장에서 잡은 패턴 1개의 정규화 위치·크기 (이후 이미지 정렬용)"""

    center_x: float = Field(description='패턴 중심 X (0~1)')
    center_y: float = Field(description='패턴 중심 Y (0~1)')
    width: float = Field(description='정규화 폭 (0~1)')
    height: float = Field(description='정규화 높이 (0~1)')
    index: int = Field(default=0, description='p0, p1, … 인덱스')

    @field_validator('center_x', 'center_y', 'width', 'height', mode='after')
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class MeasureRecipe(BaseModel):
    """
    첫 이미지 측정으로 만든 '같은 방식' 레시피.
    이후 이미지는 이 위치·개수에 맞춰 패턴을 매칭해 측정합니다.
    """

    patterns: list[PatternRef] = Field(default_factory=list)
    multi_pattern: bool = True
    match_threshold: float = Field(
        default=0.15,
        description='기준 중심과의 유클리드 거리 허용 (정규화 좌표)',
    )


class ImageExtraction(BaseModel):
    measurements: list[Measurement] = Field(default_factory=list)
    recipe: MeasureRecipe | None = Field(
        default=None,
        description='이번 측정으로 만든/사용한 레시피 (다음 이미지에 전달 가능)',
    )


# 구조 높이 대비 폭 측정 위치 (0=꼭대기, 1=바닥)
TOP_FRAC = 0.12
MID_FRAC = 0.50
BOT_FRAC = 0.88
# 하단 정보바/스케일바 제외 비율
BOTTOM_CROP = 0.88
# 측벽각 피팅에 쓸 높이 구간
ANGLE_Y0 = 0.20
ANGLE_Y1 = 0.80
# 반복 패턴으로 볼 상대 크기 허용 범위 (기준=최대 score 구조)
SIZE_RATIO_LO = 0.35
SIZE_RATIO_HI = 2.5
# 스케일바/정보바 제외 하단 비율 (overlay 검출용)
SCALEBAR_Y0 = 0.88
# overlay 주석(측정선/라벨) 픽셀 비율 임계
OVERLAY_SCORE_THR = 0.0008


def is_scalebar_measurement(
    *,
    region: str = '',
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
    orientation: str = 'horizontal',
    unit: str = '',
) -> bool:
    """스케일바/배율 표기로 보이는 측정인지 판별 (구조 측정에서 제외)."""
    region_l = (region or '').lower()
    unit_l = (unit or '').lower()
    if any(k in region_l for k in ('scale', 'bar', '배율', '스케일')):
        return True
    # 하단 코너의 짧은 가로 막대는 대개 스케일바
    if orientation == 'horizontal' and anchor_y >= SCALEBAR_Y0:
        if anchor_x <= 0.25 or anchor_x >= 0.75:
            return True
    # 단위만 있고 값이 배율처럼 쓰인 경우 (예: region에 mag)
    if 'mag' in region_l or 'x' == region_l.strip():
        return True
    if unit_l in {'x', '×'} and orientation == 'horizontal' and anchor_y >= 0.85:
        return True
    return False


def detect_has_measurement_overlay(
    image_path: str | Path,
    *,
    score_threshold: float = OVERLAY_SCORE_THR,
    exclude_bottom: float = SCALEBAR_Y0,
) -> tuple[bool, float]:
    """
    이미지에 (스케일바 제외) 측정 오버레이(녹색 선/컬러 라벨 박스)가 있는지 휴리스틱 검출.

    Returns
    -------
    (has_overlay, score)  score = 주석 픽셀 / 전체 픽셀
    """
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False, 0.0

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 측정선(녹색)·라벨 박스(청록/파랑/자홍) 대략 범위
    masks = [
        cv2.inRange(hsv, (35, 70, 70), (95, 255, 255)),   # green
        cv2.inRange(hsv, (85, 70, 70), (135, 255, 255)),  # cyan-blue
        cv2.inRange(hsv, (140, 50, 70), (175, 255, 255)), # magenta/pink arc
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # 하단 스케일바/정보바 제외
    y_cut = int(h * exclude_bottom)
    combined[y_cut:, :] = 0

    # 작은 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    score = float((combined > 0).sum()) / float(max(h * w, 1))
    return score >= score_threshold, score


def _to_gray(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'이미지 로드 실패: {image_path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _score_contour(c: np.ndarray, wh: int, ww: int) -> tuple[float, tuple[int, int, int, int]] | None:
    x, y, bw, bh = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    frame = wh * ww
    if area < frame * 0.002:
        return None
    # 배경 전체(거의 full-frame) contour 제외
    if area > frame * 0.35 or bw > ww * 0.85:
        return None
    tall = bh / max(bw, 1)
    # 기둥형(세로로 긴) 구조에 가산점
    score = area * (1.0 + 0.8 * min(tall, 4.0))
    return float(score), (x, y, bw, bh)


def _split_merged_blob(mask_roi: np.ndarray, min_gap: int = 3) -> list[np.ndarray]:
    """
    바닥이 이어져 하나의 blob이 된 반복 패턴을 세로 투영 valley로 분리.
    분리가 안 되면 원본 1개만 반환.
    """
    h, w = mask_roi.shape
    if w < 20:
        return [mask_roi]

    col_sum = (mask_roi > 0).sum(axis=0).astype(float)
    if col_sum.max() < 1:
        return [mask_roi]

    # 중앙 높이 구간만 써서 기판 연결부의 영향 감소
    y0, y1 = int(h * 0.15), int(h * 0.75)
    mid = mask_roi[y0:y1, :]
    col_mid = (mid > 0).sum(axis=0).astype(float)
    if col_mid.max() < 1:
        col_mid = col_sum

    thresh = max(col_mid.max() * 0.15, 1.0)
    binary = col_mid >= thresh

    # True 구간 = 기둥 후보
    segments: list[tuple[int, int]] = []
    i = 0
    while i < w:
        if not binary[i]:
            i += 1
            continue
        j = i
        while j < w and binary[j]:
            j += 1
        if j - i >= min_gap:
            segments.append((i, j - 1))
        i = j

    if len(segments) <= 1:
        return [mask_roi]

    # 세그먼트 폭이 너무 들쭉날쭉하면 분리 포기
    widths = [b - a + 1 for a, b in segments]
    med_w = float(np.median(widths))
    keep = [(a, b) for (a, b), ww in zip(segments, widths) if 0.4 * med_w <= ww <= 2.5 * med_w]
    if len(keep) <= 1:
        return [mask_roi]

    out = []
    for a, b in keep:
        pad = max(1, (b - a) // 10)
        x0 = max(0, a - pad)
        x1 = min(w, b + pad + 1)
        part = np.zeros_like(mask_roi)
        part[:, x0:x1] = mask_roi[:, x0:x1]
        if (part > 0).sum() > 0:
            out.append(part)
    return out if out else [mask_roi]


def _contours_from_threshold(th: np.ndarray) -> list[tuple[float, np.ndarray, tuple[int, int, int, int]]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wh, ww = th.shape
    scored: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for c in cnts:
        res = _score_contour(c, wh, ww)
        if res is None:
            continue
        score, bbox = res
        scored.append((score, c, bbox))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _estimate_structure_instances(
    gray: np.ndarray,
    *,
    multi_pattern: bool = True,
    max_instances: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    """
    구조 마스크 인스턴스 목록 (왼쪽→오른쪽 정렬)과 공통 meta 반환.
    multi_pattern=False 이면 score 최대 1개만.
    """
    h, w = gray.shape
    work = gray[: int(h * BOTTOM_CROP), :].copy()
    wh, ww = work.shape

    blur = cv2.GaussianBlur(work, (5, 5), 0)
    _, th_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    s1 = _contours_from_threshold(th_bright)
    s2 = _contours_from_threshold(th_dark)

    def polarity_score(scored: list) -> float:
        """단일 최고점 + 반복 후보 개수 보너스"""
        if not scored:
            return -1.0
        top = scored[0][0]
        ref_bw, ref_bh = scored[0][2][2], scored[0][2][3]
        ref_area = max(ref_bw * ref_bh, 1)
        n_sim = 0
        for _sc, _c, bb in scored[:12]:
            area = max(bb[2] * bb[3], 1)
            if (
                SIZE_RATIO_LO <= area / ref_area <= SIZE_RATIO_HI
                and SIZE_RATIO_LO <= bb[3] / max(ref_bh, 1) <= SIZE_RATIO_HI
            ):
                n_sim += 1
        return top * (1.0 + 0.35 * max(0, n_sim - 1))

    scored = s1 if polarity_score(s1) >= polarity_score(s2) else s2

    meta: dict = {
        'full_h': h,
        'full_w': w,
        'work_h': wh,
        'work_w': ww,
    }
    if not scored:
        return [], meta

    # 기준(최고 score)과 비슷한 크기만 반복 패턴으로 채택
    _ref_score, _ref_c, ref_bbox = scored[0]
    _, _, ref_bw, ref_bh = ref_bbox
    ref_area = max(ref_bw * ref_bh, 1)

    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for score, c, bbox in scored:
        if not multi_pattern and candidates:
            break
        _x, _y, bw, bh = bbox
        area = max(bw * bh, 1)
        area_ok = SIZE_RATIO_LO <= (area / ref_area) <= SIZE_RATIO_HI
        h_ok = SIZE_RATIO_LO <= (bh / max(ref_bh, 1)) <= SIZE_RATIO_HI
        if area_ok and h_ok:
            candidates.append((score, c, bbox))

    if not candidates:
        candidates = [scored[0]]

    # 마스크 생성 + 붙은 blob 분리 시도
    masks: list[np.ndarray] = []
    for _score, c, bbox in candidates:
        m = np.zeros((wh, ww), dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=-1)
        x, y, bw, bh = bbox
        if multi_pattern and bw >= 40:
            roi = m[y : y + bh, x : x + bw]
            parts = _split_merged_blob(roi)
            if len(parts) > 1:
                for part in parts:
                    full = np.zeros_like(m)
                    full[y : y + bh, x : x + bw] = part
                    masks.append(full)
                continue
        masks.append(m)

    # 중복(거의 같은 위치) 제거: 중심 x 기준
    dedup: list[tuple[float, np.ndarray]] = []
    for m in masks:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        if any(abs(cx - pcx) < 8 for pcx, _ in dedup):
            continue
        dedup.append((cx, m))
    dedup.sort(key=lambda t: t[0])

    if max_instances is not None and max_instances > 0:
        dedup = dedup[:max_instances]

    instances = [m for _, m in dedup]
    return instances, meta


def _instance_norm_box(
    mask: np.ndarray, full_w: int, full_h: int
) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = ((x0 + x1) / 2) / full_w
    cy = ((y0 + y1) / 2) / full_h
    bw = max(x1 - x0, 1.0) / full_w
    bh = max(y1 - y0, 1.0) / full_h
    return cx, cy, bw, bh


def build_recipe_from_geoms(
    geoms: list[dict],
    *,
    full_w: int,
    full_h: int,
    multi_pattern: bool = True,
    match_threshold: float = 0.15,
) -> MeasureRecipe:
    """측정 기하정보 → 이후 이미지용 레시피"""
    patterns: list[PatternRef] = []
    for i, g in enumerate(geoms):
        cx = ((g['x0'] + g['x1']) / 2) / full_w
        cy = ((g['y0'] + g['y1']) / 2) / full_h
        bw = max(g['x1'] - g['x0'], 1) / full_w
        bh = max(g['y1'] - g['y0'], 1) / full_h
        patterns.append(
            PatternRef(
                center_x=float(cx),
                center_y=float(cy),
                width=float(bw),
                height=float(bh),
                index=i,
            )
        )
    return MeasureRecipe(
        patterns=patterns,
        multi_pattern=multi_pattern and len(patterns) > 1,
        match_threshold=match_threshold,
    )


def build_recipe_from_measurements(
    measurements: list[Measurement],
    *,
    multi_pattern: bool | None = None,
    match_threshold: float = 0.15,
) -> MeasureRecipe:
    """
    Measurement 리스트에서 레시피 추정.
    height(또는 *_height) 항목의 앵커를 패턴 중심으로 사용.
    """
    heights = [
        m
        for m in measurements
        if m.orientation == 'vertical'
        and (m.region == 'height' or m.region.endswith('_height'))
    ]
    heights.sort(key=lambda m: m.anchor_x)
    patterns = [
        PatternRef(
            center_x=m.anchor_x,
            center_y=m.anchor_y,
            width=0.08,
            height=0.35,
            index=i,
        )
        for i, m in enumerate(heights)
    ]
    if multi_pattern is None:
        multi_pattern = len(patterns) > 1
    return MeasureRecipe(
        patterns=patterns,
        multi_pattern=bool(multi_pattern),
        match_threshold=match_threshold,
    )


def _match_instances_to_recipe(
    instances: list[np.ndarray],
    recipe: MeasureRecipe,
    *,
    full_w: int,
    full_h: int,
) -> list[np.ndarray | None]:
    """
    레시피 패턴 순서대로 가장 가까운 인스턴스를 할당.
    매칭 실패 시 해당 슬롯은 None.
    """
    boxes: list[tuple[int, float, float, np.ndarray]] = []
    for i, mask in enumerate(instances):
        box = _instance_norm_box(mask, full_w, full_h)
        if box is None:
            continue
        cx, cy, _bw, _bh = box
        boxes.append((i, cx, cy, mask))

    used: set[int] = set()
    ordered: list[np.ndarray | None] = []
    thr = float(recipe.match_threshold)

    for pref in recipe.patterns:
        best_j = None
        best_d = 1e9
        for j, cx, cy, _mask in boxes:
            if j in used:
                continue
            d = float(np.hypot(cx - pref.center_x, cy - pref.center_y))
            # 크기 차이 페널티 (너무 다른 blob 회피)
            box = _instance_norm_box(_mask, full_w, full_h)
            if box is not None:
                _cx, _cy, bw, bh = box
                size_pen = 0.5 * (
                    abs(bw - pref.width) / max(pref.width, 0.01)
                    + abs(bh - pref.height) / max(pref.height, 0.01)
                )
                d = d + 0.05 * size_pen
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None or best_d > thr:
            ordered.append(None)
            continue
        used.add(best_j)
        mask = next(m for j, _cx, _cy, m in boxes if j == best_j)
        ordered.append(mask)
    return ordered


def _edge_profiles(
    mask: np.ndarray, y0: int, y1: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """각 y에서 좌/우 edge x 좌표"""
    ys, lefts, rights = [], [], []
    for y in range(y0, y1 + 1):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < 2:
            continue
        ys.append(y)
        lefts.append(int(xs.min()))
        rights.append(int(xs.max()))
    return np.array(ys), np.array(lefts), np.array(rights)


def _smooth(arr: np.ndarray, k: int = 7) -> np.ndarray:
    if len(arr) < k:
        return arr.astype(float)
    ker = np.ones(k) / k
    return np.convolve(arr.astype(float), ker, mode='same')


def _measure_one_mask(
    mask: np.ndarray,
    *,
    full_w: int,
    full_h: int,
    scale: float,
    unit: str,
    region_prefix: str = '',
) -> tuple[list[Measurement], dict | None]:
    """단일 마스크 → 측정 리스트 + 오버레이용 기하정보"""
    ys_pix, xs_pix = np.where(mask > 0)
    if len(xs_pix) == 0:
        return [], None

    y0, y1 = int(ys_pix.min()), int(ys_pix.max())
    x0, x1 = int(xs_pix.min()), int(xs_pix.max())
    ys, lefts, rights = _edge_profiles(mask, y0, y1)
    if len(ys) < 10:
        return [], None

    lefts_s = _smooth(lefts)
    rights_s = _smooth(rights)
    widths = rights_s - lefts_s
    height_px = float(y1 - y0)
    if height_px < 5:
        return [], None

    def width_at_frac(frac: float) -> tuple[float, float, float]:
        y_target = y0 + frac * (y1 - y0)
        idx = int(np.argmin(np.abs(ys.astype(float) - y_target)))
        w = float(widths[idx])
        cx = float((lefts_s[idx] + rights_s[idx]) / 2)
        cy = float(ys[idx])
        return w, cx, cy

    w_top, cx_t, cy_t = width_at_frac(TOP_FRAC)
    w_mid, cx_m, cy_m = width_at_frac(MID_FRAC)
    w_bot, cx_b, cy_b = width_at_frac(BOT_FRAC)

    mid_mask = (ys >= y0 + ANGLE_Y0 * height_px) & (ys <= y0 + ANGLE_Y1 * height_px)
    angle_deg = None
    if mid_mask.sum() >= 5:
        yy = ys[mid_mask].astype(float)
        xx = rights_s[mid_mask]
        a, _b = np.polyfit(yy, xx, 1)
        if abs(a) > 1e-6:
            angle_from_vertical = np.degrees(np.arctan(a))
            angle_deg = float(90.0 - abs(angle_from_vertical))
        else:
            angle_deg = 90.0

    def norm_x(x: float) -> float:
        return float(np.clip(x / full_w, 0, 1))

    def norm_y(y: float) -> float:
        return float(np.clip(y / full_h, 0, 1))

    def rname(base: str) -> str:
        return f'{region_prefix}{base}' if region_prefix else base

    items: list[Measurement] = [
        Measurement(
            value=round(height_px * scale, 3),
            unit=unit,
            anchor_x=norm_x((x0 + x1) / 2),
            anchor_y=norm_y((y0 + y1) / 2),
            orientation='vertical',
            region=rname('height'),
        ),
        Measurement(
            value=round(w_top * scale, 3),
            unit=unit,
            anchor_x=norm_x(cx_t),
            anchor_y=norm_y(cy_t),
            orientation='horizontal',
            region=rname('top'),
        ),
        Measurement(
            value=round(w_mid * scale, 3),
            unit=unit,
            anchor_x=norm_x(cx_m),
            anchor_y=norm_y(cy_m),
            orientation='horizontal',
            region=rname('middle'),
        ),
        Measurement(
            value=round(w_bot * scale, 3),
            unit=unit,
            anchor_x=norm_x(cx_b),
            anchor_y=norm_y(cy_b),
            orientation='horizontal',
            region=rname('bottom'),
        ),
    ]
    if angle_deg is not None:
        items.append(
            Measurement(
                value=round(angle_deg, 2),
                unit='°',
                anchor_x=norm_x(float(rights_s[len(rights_s) // 2])),
                anchor_y=norm_y(float(ys[len(ys) // 2])),
                orientation='angle',
                region=rname('right_sidewall'),
            )
        )

    geom = {
        'y0': y0,
        'y1': y1,
        'x0': x0,
        'x1': x1,
        'ys': ys,
        'lefts_s': lefts_s,
        'rights_s': rights_s,
        'cy_t': cy_t,
        'cx_t': cx_t,
        'cy_m': cy_m,
        'cx_m': cx_m,
        'cy_b': cy_b,
        'cx_b': cx_b,
        'angle_deg': angle_deg,
        'items': items,
    }
    return items, geom


def measure_sem_structure(
    image_path: str | Path,
    *,
    nm_per_pixel: float | None = None,
    save_overlay: bool = True,
    overlay_dir: str | Path | None = None,
    multi_pattern: bool = True,
    max_instances: int | None = None,
    recipe: MeasureRecipe | None = None,
    match_threshold: float | None = None,
) -> ImageExtraction:
    """
    오버레이 없는 SEM 단면에서 구조 edge를 잡아 측정.

    Parameters
    ----------
    image_path : 이미지 경로
    nm_per_pixel : 픽셀당 nm. None이면 길이 단위는 px
    save_overlay : edge/폭 위치 시각화 저장 여부
    overlay_dir : 오버레이 저장 폴더 (None이면 data/metrology/output/edge_overlay)
    multi_pattern : True면 반복 패턴을 인스턴스별로 모두 측정 (왼쪽→오른쪽 p0, p1, …)
    max_instances : 최대 인스턴스 수 (None이면 제한 없음)
    recipe : 첫 장에서 만든 MeasureRecipe. 있으면 같은 위치·개수로 매칭 측정
    match_threshold : recipe 매칭 거리 허용치 (None이면 recipe.match_threshold)

    Returns
    -------
    ImageExtraction : measurements + recipe
        첫 장이면 새 recipe 생성, recipe 입력이 있으면 그걸 그대로 반환
    """
    path = Path(image_path)
    gray = _to_gray(path)

    use_multi = multi_pattern if recipe is None else recipe.multi_pattern
    use_max = max_instances
    if recipe is not None and recipe.patterns:
        use_max = max(len(recipe.patterns) * 3, len(recipe.patterns))
        # 후보를 넉넉히 찾은 뒤 recipe로 고름
        use_multi = True

    instances, meta = _estimate_structure_instances(
        gray,
        multi_pattern=use_multi,
        max_instances=use_max,
    )
    if not instances:
        print(f'[edge] 구조 마스크 실패: {path.name}')
        return ImageExtraction(measurements=[], recipe=recipe)

    full_w, full_h = meta['full_w'], meta['full_h']

    active_recipe = recipe
    if recipe is not None and recipe.patterns:
        thr = (
            float(match_threshold)
            if match_threshold is not None
            else float(recipe.match_threshold)
        )
        recipe_use = recipe.model_copy(update={'match_threshold': thr})
        matched = _match_instances_to_recipe(
            instances, recipe_use, full_w=full_w, full_h=full_h
        )
        n_ok = sum(1 for m in matched if m is not None)
        print(
            f'[edge] recipe match: {n_ok}/{len(recipe.patterns)} '
            f'(threshold={thr:.3f}) - {path.name}'
        )
        # 매칭 실패한 슬롯은 건너뛰되 인덱스는 유지하기 위해 None 허용
        instances_ordered: list[tuple[int, np.ndarray]] = [
            (i, m) for i, m in enumerate(matched) if m is not None
        ]
        if not instances_ordered:
            print(f'[edge] recipe 매칭 실패, 자동 검출로 fallback: {path.name}')
            instances_ordered = list(enumerate(instances))
            active_recipe = None
        else:
            # 레시피 인덱스(p0,p1,…) 유지
            pass
    else:
        instances_ordered = list(enumerate(instances))

    unit = 'nm' if nm_per_pixel and nm_per_pixel > 0 else 'px'
    scale = float(nm_per_pixel) if unit == 'nm' else 1.0

    # region prefix: recipe 있으면 항상 p{i}_ (패턴 1개여도 일관성),
    # 없으면 기존처럼 2개 이상일 때만
    n_label = (
        len(recipe.patterns)
        if recipe is not None and recipe.patterns
        else len(instances_ordered)
    )
    use_prefix = n_label > 1 or (
        recipe is not None and len(getattr(recipe, 'patterns', [])) > 1
    )

    all_items: list[Measurement] = []
    geoms: list[dict] = []
    geom_indices: list[int] = []

    for idx, mask in instances_ordered:
        prefix = f'p{idx}_' if use_prefix else ''
        items, geom = _measure_one_mask(
            mask,
            full_w=full_w,
            full_h=full_h,
            scale=scale,
            unit=unit,
            region_prefix=prefix,
        )
        if not items or geom is None:
            print(f'[edge] edge 포인트 부족 (instance {idx}): {path.name}')
            continue
        all_items.extend(items)
        geoms.append(geom)
        geom_indices.append(idx)

    if not all_items:
        return ImageExtraction(measurements=[], recipe=recipe)

    # 첫 장이면 recipe 생성
    if active_recipe is None and recipe is None:
        active_recipe = build_recipe_from_geoms(
            geoms,
            full_w=full_w,
            full_h=full_h,
            multi_pattern=use_multi and len(geoms) > 1,
            match_threshold=(
                float(match_threshold) if match_threshold is not None else 0.15
            ),
        )
    elif recipe is not None:
        active_recipe = recipe

    if save_overlay:
        if overlay_dir is None:
            overlay_dir = Path('data') / 'metrology' / 'output' / 'edge_overlay'
        out_dir = Path(overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        palette = [
            (0, 255, 0),
            (255, 128, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 255, 128),
            (255, 255, 0),
        ]
        for gi, g in enumerate(geoms):
            i = geom_indices[gi]
            color = palette[i % len(palette)]
            ys, lefts_s, rights_s = g['ys'], g['lefts_s'], g['rights_s']
            for y, L, R in zip(ys[::2], lefts_s[::2], rights_s[::2]):
                cv2.circle(vis, (int(L), int(y)), 1, color, -1)
                cv2.circle(vis, (int(R), int(y)), 1, color, -1)
            for cy, cx, bar in [
                (g['cy_t'], g['cx_t'], (255, 0, 0)),
                (g['cy_m'], g['cx_m'], (0, 255, 255)),
                (g['cy_b'], g['cx_b'], (0, 128, 255)),
            ]:
                cv2.line(vis, (int(cx - 20), int(cy)), (int(cx + 20), int(cy)), bar, 1)
            cv2.rectangle(vis, (g['x0'], g['y0']), (g['x1'], g['y1']), color, 1)
            label = f'p{i}' if use_prefix else 'p0'
            cv2.putText(
                vis,
                label,
                (g['x0'], max(12, g['y0'] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        out_path = out_dir / f'{path.stem}_edge.png'
        cv2.imwrite(str(out_path), vis)
        print(f'[edge] overlay → {out_path}')

    # 요약 로그
    if use_prefix:
        print(f'[edge] {path.name}: {len(geoms)}개 패턴 측정')
        for gi, g in enumerate(geoms):
            i = geom_indices[gi]
            its = g['items']
            ang = g['angle_deg']
            print(
                f'  p{i}: H={its[0].value}{unit}, '
                f'W(top/mid/bot)={its[1].value}/{its[2].value}/{its[3].value}{unit}'
                + (f', angle={ang:.1f}°' if ang is not None else '')
            )
    else:
        items = geoms[0]['items']
        ang = geoms[0]['angle_deg']
        print(
            f'[edge] {path.name}: H={items[0].value}{unit}, '
            f'W(top/mid/bot)={items[1].value}/{items[2].value}/{items[3].value}{unit}'
            + (f', angle={ang:.1f}°' if ang is not None else '')
        )

    return ImageExtraction(measurements=all_items, recipe=active_recipe)


def measure_sem_batch(
    image_paths: list[str | Path],
    *,
    nm_per_pixel: float | None = None,
    save_overlay: bool = True,
    overlay_dir: str | Path | None = None,
    multi_pattern: bool = True,
    max_instances: int | None = None,
    match_threshold: float = 0.15,
) -> tuple[list[ImageExtraction], MeasureRecipe | None]:
    """
    첫 장으로 레시피를 만들고, 나머지 이미지는 같은 방식으로 매칭 측정.

    Returns
    -------
    (결과 리스트, 첫 장 recipe)
    """
    results: list[ImageExtraction] = []
    recipe: MeasureRecipe | None = None
    paths = [Path(p) for p in image_paths]
    if not paths:
        return results, None

    first = measure_sem_structure(
        paths[0],
        nm_per_pixel=nm_per_pixel,
        save_overlay=save_overlay,
        overlay_dir=overlay_dir,
        multi_pattern=multi_pattern,
        max_instances=max_instances,
        match_threshold=match_threshold,
    )
    results.append(first)
    recipe = first.recipe
    if recipe is not None:
        print(
            f'[edge] 기준 레시피 확정: {len(recipe.patterns)}개 패턴 '
            f'(threshold={recipe.match_threshold})'
        )

    for p in paths[1:]:
        results.append(
            measure_sem_structure(
                p,
                nm_per_pixel=nm_per_pixel,
                save_overlay=save_overlay,
                overlay_dir=overlay_dir,
                multi_pattern=multi_pattern,
                max_instances=max_instances,
                recipe=recipe,
                match_threshold=match_threshold,
            )
        )
    return results, recipe
