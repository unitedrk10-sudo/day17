"""
SEM/TEM 스케일바 · 배율 읽기 Tool.

역할
----
1) 이미지에서 스케일바/정보바 영역을 찾고
2) 막대 길이(px) + 표기 길이(nm/µm 등)로 nm_per_pixel 계산
3) 배율: 이미지에 표기가 **없는 경우가 많음** → 스케일바 nm/px 로 상대 배율 추정
   (표기가 있으면 절대값 사용, ``max_only`` 비교는 추정값으로도 가능)

method
------
- ``cv``    : 하단 밴드 가로 막대 검출 + 텍스트 휴리스틱(라벨이 이미지에 박혀 있을 때 약함)
- ``vlm``   : gpt-4o 로 스케일바/nm_per_px 추출 (권장)
- ``auto``  : CV로 후보 찾은 뒤, 텍스트·스케일이 비면 VLM 보강

Usage
-----
    from tools.scale_bar import read_scale_bar, filter_by_mag

    r = read_scale_bar('sample.png', method='auto')
    print(r.nm_per_pixel, r.magnification, r.mag_source, r.kind)
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# 정보바/스케일바가 주로 있는 하단 비율
INFO_BAND_Y0 = 0.82
# 너무 짧은/긴 막대 제외 (이미지 폭 대비)
BAR_WIDTH_FRAC_MIN = 0.03
BAR_WIDTH_FRAC_MAX = 0.45
BAR_ASPECT_MIN = 4.0  # width/height

UNIT_TO_NM = {
    'nm': 1.0,
    '나노': 1.0,
    'um': 1e3,
    'µm': 1e3,
    'μm': 1e3,
    'micron': 1e3,
    'microns': 1e3,
    'mm': 1e6,
    'å': 0.1,
    'a': 0.1,
    'angstrom': 0.1,
}


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------

ScaleKind = Literal['classic', 'info_bar', 'derived_overlay', 'none']
ReadMethod = Literal['auto', 'cv', 'vlm']


MagSource = Literal['printed', 'scale_estimated', 'none']


class ScaleBarReading(BaseModel):
    """한 장 스케일바/배율 읽기 결과."""

    image_path: str
    found: bool = False
    kind: ScaleKind = 'none'
    bar_px: float | None = None
    label_value: float | None = None
    label_unit: str | None = None
    nm_per_pixel: float | None = None
    magnification: float | None = Field(
        default=None,
        description=(
            '비교용 배율. 이미지에 50kx 등 표기가 있으면 절대값, '
            '없으면 스케일바 nm/px 로 상대 추정 (max_only 정렬용)'
        ),
    )
    mag_source: MagSource = 'none'
    mag_label: str | None = None
    # 정규화 박스 (x0,y0,x1,y1) in 0~1 — 스케일바/정보바 ROI
    region_xyxy: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    raw_text: str = ''
    method_used: str = ''
    warnings: list[str] = Field(default_factory=list)

    @field_validator('confidence', mode='after')
    @classmethod
    def clamp_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class ScaleBarBatchResult(BaseModel):
    readings: list[ScaleBarReading]
    scale_map: dict[str, float] = Field(
        default_factory=dict,
        description='path → nm_per_pixel (성공분만)',
    )
    mag_map: dict[str, float] = Field(
        default_factory=dict,
        description='path → magnification (성공분만)',
    )


# ---------------------------------------------------------------------------
# 텍스트 파서
# ---------------------------------------------------------------------------

_SCALE_PATTERNS = [
    # 100 nm, 2.5µm, 1 um, 500Å
    re.compile(
        r'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>nm|µm|μm|um|mm|å|Å|angstroms?|microns?)',
        re.IGNORECASE,
    ),
]

_MAG_PATTERNS = [
    # 50.0kx, 50 kx, 50.00kX
    re.compile(
        r'(?P<val>\d+(?:\.\d+)?)\s*[kK]\s*[xX×]',
    ),
    # Mag: 50000x , ×50,000 , x 50000
    re.compile(
        r'(?:mag(?:nification)?\s*[:=]?\s*)?(?P<val>\d{2,3}(?:,\d{3})+|\d{4,7})\s*[xX×]',
        re.IGNORECASE,
    ),
    # 50000×
    re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*[xX×]\b'),
]


def unit_to_nm_factor(unit: str) -> float | None:
    u = (unit or '').strip().lower().replace('μ', 'µ')
    if u in UNIT_TO_NM:
        return UNIT_TO_NM[u]
    if u.startswith('micron'):
        return 1e3
    if u.startswith('angstrom') or u in {'å', 'Å'.lower()}:
        return 0.1
    return None


def parse_scale_label(text: str) -> tuple[float | None, str | None]:
    """문장에서 스케일 표기 (값, 단위) 추출. 첫 매칭."""
    if not text:
        return None, None
    for pat in _SCALE_PATTERNS:
        m = pat.search(text)
        if m:
            unit = m.group('unit')
            # 정규화
            ul = unit.lower().replace('μ', 'µ')
            if ul in {'µm', 'μm', 'um', 'micron', 'microns'}:
                unit_n = 'µm'
            elif ul in {'å', 'Å'.lower()} or ul.startswith('angstrom'):
                unit_n = 'Å'
            else:
                unit_n = ul
            return float(m.group('val')), unit_n
    return None, None


def parse_magnification(text: str) -> tuple[float | None, str | None]:
    """
    배율 숫자로 환산 (이미지에 **표기가 있을 때만**).
    50.0kx → 50000, 50000x → 50000

    운영 이미지에는 배율 숫자가 없는 경우가 많다 → ``estimate_mag_from_scale``.
    """
    if not text:
        return None, None
    # kx 우선
    m = _MAG_PATTERNS[0].search(text)
    if m:
        val = float(m.group('val')) * 1000.0
        return val, m.group(0).strip()
    for pat in _MAG_PATTERNS[1:]:
        m = pat.search(text)
        if m:
            raw = m.group('val').replace(',', '')
            val = float(raw)
            # 너무 작은 값(예: 10x)은 광학 배율일 수 있음 — SEM이면 보통 >= 100
            return val, m.group(0).strip()
    return None, None


def estimate_mag_from_scale(
    nm_per_pixel: float,
    *,
    ref_nm_per_pixel: float = 1.0,
) -> float:
    """
    스케일바 ``nm/px`` → 비교용 상대 배율.

    nm/px 가 작을수록 FOV가 좁고 배율이 크다.
    ``mag ∝ 1 / nm_per_pixel`` (절대 Mag 숫자는 센서 정보가 없어 알 수 없음).
    """
    if nm_per_pixel <= 0:
        raise ValueError('nm_per_pixel > 0 필요')
    return float(ref_nm_per_pixel) / float(nm_per_pixel)


def effective_magnification(reading: ScaleBarReading) -> float | None:
    """표기 배율 우선, 없으면 스케일 추정 배율."""
    if reading.magnification is not None and reading.magnification > 0:
        return float(reading.magnification)
    if reading.nm_per_pixel is not None and reading.nm_per_pixel > 0:
        return estimate_mag_from_scale(reading.nm_per_pixel)
    return None


def apply_scale_estimated_mag(reading: ScaleBarReading) -> ScaleBarReading:
    """
    배율 표기가 없으면 스케일바 nm/px 로 상대 배율을 채운다.

    이미 표기 배율이 있으면 그대로 둔다.
    """
    if reading.magnification is not None and reading.magnification > 0:
        if reading.mag_source == 'none':
            reading.mag_source = 'printed'
        return reading
    if reading.nm_per_pixel is not None and reading.nm_per_pixel > 0:
        est = estimate_mag_from_scale(reading.nm_per_pixel)
        reading.magnification = est
        reading.mag_source = 'scale_estimated'
        reading.mag_label = (
            reading.mag_label
            or f'~rel from {reading.nm_per_pixel:.4g} nm/px'
        )
        msg = '배율 표기 없음 → 스케일바 nm/px로 상대 배율 추정'
        if msg not in reading.warnings:
            reading.warnings.append(msg)
    return reading


def physical_nm(value: float, unit: str) -> float | None:
    fac = unit_to_nm_factor(unit)
    if fac is None:
        return None
    return float(value) * fac


# ---------------------------------------------------------------------------
# CV: 하단 밴드 · 가로 막대
# ---------------------------------------------------------------------------

def _load_bgr(image_path: str | Path) -> np.ndarray:
    """Unicode 경로 안전 로드 (Windows 한글 경로 대응)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f'이미지 없음: {path}')
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'이미지 디코드 실패: {path}')
    return img


def _imwrite_unicode(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else '.png'
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f'이미지 인코드 실패: {path}')
    buf.tofile(str(path))


def crop_info_band(
    bgr: np.ndarray,
    *,
    y0_frac: float = INFO_BAND_Y0,
) -> tuple[np.ndarray, int]:
    """하단 정보바 밴드와 y0 픽셀 offset."""
    h = bgr.shape[0]
    y0 = int(h * y0_frac)
    return bgr[y0:, :, :].copy(), y0


def _find_bar_candidates(
    band_bgr: np.ndarray,
    *,
    full_width: int,
) -> list[dict[str, Any]]:
    """밝거나 대비 큰 가로 막대 후보."""
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    bh, bw = gray.shape[:2]
    # 이진화: 밝은 막대 / 어두운 막대 둘 다
    outs: list[dict[str, Any]] = []
    for invert in (False, True):
        if invert:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, bw // 80), 1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h < 1 or w < 1:
                continue
            aspect = w / float(h)
            w_frac = w / float(max(full_width, 1))
            if aspect < BAR_ASPECT_MIN:
                continue
            if not (BAR_WIDTH_FRAC_MIN <= w_frac <= BAR_WIDTH_FRAC_MAX):
                continue
            if h > bh * 0.45:
                continue
            outs.append(
                {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'aspect': aspect,
                    'w_frac': w_frac,
                    'score': aspect * w_frac,
                    'invert': invert,
                }
            )
    outs.sort(key=lambda d: d['score'], reverse=True)
    # 중복 제거 (IoU)
    kept: list[dict[str, Any]] = []
    for cand in outs:
        ok = True
        for k in kept:
            if _iou_1d(cand['x'], cand['w'], k['x'], k['w']) > 0.5 and abs(cand['y'] - k['y']) < 8:
                ok = False
                break
        if ok:
            kept.append(cand)
    return kept[:5]


def _iou_1d(x0: int, w0: int, x1: int, w1: int) -> float:
    a0, a1 = x0, x0 + w0
    b0, b1 = x1, x1 + w1
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def read_scale_bar_cv(
    image_path: str | Path,
    *,
    y0_frac: float = INFO_BAND_Y0,
    known_label: tuple[float, str] | None = None,
) -> ScaleBarReading:
    """
    CV만으로 하단 스케일바 막대를 찾고 nm/px를 계산.

    known_label : (100, 'nm')처럼 표기 길이를 이미 알 때 사용.
    텍스트 OCR은 포함하지 않음 → 표기 없으면 warnings에 남김.
    """
    path = Path(image_path)
    bgr = _load_bgr(path)
    h, w = bgr.shape[:2]
    band, y0 = crop_info_band(bgr, y0_frac=y0_frac)
    cands = _find_bar_candidates(band, full_width=w)

    warnings: list[str] = []
    if not cands:
        return ScaleBarReading(
            image_path=str(path),
            found=False,
            kind='none',
            method_used='cv',
            warnings=['하단 밴드에서 가로 스케일바 막대를 찾지 못함'],
        )

    best = cands[0]
    bar_px = float(best['w'])
    region = (
        best['x'] / w,
        (y0 + best['y']) / h,
        (best['x'] + best['w']) / w,
        (y0 + best['y'] + best['h']) / h,
    )

    label_value = None
    label_unit = None
    nm_per_px = None
    if known_label:
        label_value, label_unit = known_label
        phys = physical_nm(label_value, label_unit)
        if phys and bar_px > 0:
            nm_per_px = phys / bar_px
    else:
        warnings.append('CV는 막대만 검출. 표기 길이는 VLM/OCR 또는 known_label 필요')

    # 정보바 전체인지(두꺼운 하단 띠) vs 단독 스케일바
    kind: ScaleKind = 'classic'
    band_h_frac = band.shape[0] / float(h)
    if band_h_frac >= 0.12 and best['w_frac'] < 0.2:
        kind = 'info_bar'

    conf = min(1.0, 0.35 + 0.1 * best['aspect'] / 10.0 + (0.4 if nm_per_px else 0.0))

    return apply_scale_estimated_mag(
        ScaleBarReading(
            image_path=str(path),
            found=True,
            kind=kind,
            bar_px=bar_px,
            label_value=label_value,
            label_unit=label_unit,
            nm_per_pixel=nm_per_px,
            region_xyxy=region,
            confidence=conf,
            method_used='cv',
            warnings=warnings,
        )
    )


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

_VLM_PROMPT = """이 현미경(SEM/TEM) 이미지의 스케일바 정보를 JSON만으로 답하세요.

규칙:
1) 하단에 단독 스케일바(짧은 가로 막대 + "100 nm" 같은 표기)가 있으면 kind="classic"
2) SEM 정보바(스케일·HV·WD 등이 함께 있는 하단 밴드)면 kind="info_bar"
3) 전통 스케일바는 없고, Top Width: 25.9nm 같은 측정 오버레이만 있으면
   가능하면 측정선 길이와 표기값으로 nm_per_pixel을 추정하고 kind="derived_overlay"
4) 스케일도 추정 불가하면 kind="none", found=false
5) magnification: 이미지에 "50.0kx" / "50000x"처럼 **배율 숫자가 실제로 보일 때만** 절대배수로 채운다.
   배율 표기가 없으면 반드시 null (추정하지 말 것 — 코드가 스케일바 nm/px로 상대 배율을 계산한다)
6) bar_px는 스케일바 막대의 픽셀 폭. 모르면 null
7) region_xyxy는 스케일바 또는 정보바 박스 [x0,y0,x1,y1] (0~1 정규화). 없으면 null
8) nm_per_pixel이 핵심이다. 스케일 막대+라벨(또는 오버레이)로 반드시 구하라.

반드시 이 스키마의 JSON만 출력:
{
  "found": true,
  "kind": "classic"|"info_bar"|"derived_overlay"|"none",
  "bar_px": number|null,
  "label_value": number|null,
  "label_unit": "nm"|"µm"|"mm"|"Å"|null,
  "nm_per_pixel": number|null,
  "magnification": number|null,
  "mag_label": string|null,
  "region_xyxy": [x0,y0,x1,y1]|null,
  "confidence": 0.0,
  "raw_text": "이미지에서 읽은 관련 문자열",
  "notes": "짧은 설명"
}
"""


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('utf-8')


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
    }.get(ext, 'image/png')


def _get_openai_client():
    from openai import OpenAI

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY 가 없습니다 (.env)')
    return OpenAI(api_key=key)


def read_scale_bar_vlm(
    image_path: str | Path,
    *,
    client=None,
    model: str = 'gpt-4o',
    detail: str = 'high',
) -> ScaleBarReading:
    """gpt-4o Vision으로 스케일바/배율/nm_per_pixel 읽기."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    client = client or _get_openai_client()
    b64 = _encode_image_b64(path)
    mime = _guess_mime(path)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={'type': 'json_object'},
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': _VLM_PROMPT},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime};base64,{b64}',
                            'detail': detail,
                        },
                    },
                ],
            }
        ],
    )
    raw = resp.choices[0].message.content or '{}'
    data = json.loads(raw)

    kind = data.get('kind') or 'none'
    if kind not in {'classic', 'info_bar', 'derived_overlay', 'none'}:
        kind = 'none'

    label_value = data.get('label_value')
    label_unit = data.get('label_unit')
    bar_px = data.get('bar_px')
    nm_per_px = data.get('nm_per_pixel')

    # 일관성: 값·단위·막대폭이 있으면 nm/px 재계산
    warnings: list[str] = []
    if label_value is not None and label_unit and bar_px and float(bar_px) > 0:
        phys = physical_nm(float(label_value), str(label_unit))
        if phys is not None:
            recomputed = phys / float(bar_px)
            if nm_per_px is None:
                nm_per_px = recomputed
            elif abs(float(nm_per_px) - recomputed) / max(recomputed, 1e-9) > 0.15:
                warnings.append(
                    f'VLM nm_per_pixel({nm_per_px})와 재계산({recomputed:.4f}) 차이>15% → 재계산 사용'
                )
                nm_per_px = recomputed

    # raw_text에 스케일/배율 파서 보강
    raw_text = str(data.get('raw_text') or '')
    mag = data.get('magnification')
    mag_label = data.get('mag_label')
    if mag is None and raw_text:
        mag, mag_label2 = parse_magnification(raw_text)
        mag_label = mag_label or mag_label2
    if (label_value is None or label_unit is None) and raw_text:
        pv, pu = parse_scale_label(raw_text)
        label_value = label_value if label_value is not None else pv
        label_unit = label_unit or pu

    region = data.get('region_xyxy')
    region_t = None
    if isinstance(region, (list, tuple)) and len(region) == 4:
        region_t = tuple(float(x) for x in region)  # type: ignore[assignment]

    found = bool(data.get('found')) and kind != 'none'
    if nm_per_px is None and not found:
        warnings.append(str(data.get('notes') or '스케일 정보 없음'))

    mag_f = float(mag) if mag is not None else None
    reading = ScaleBarReading(
        image_path=str(path),
        found=found or nm_per_px is not None,
        kind=kind if (found or nm_per_px is not None) else 'none',
        bar_px=float(bar_px) if bar_px is not None else None,
        label_value=float(label_value) if label_value is not None else None,
        label_unit=str(label_unit) if label_unit else None,
        nm_per_pixel=float(nm_per_px) if nm_per_px is not None else None,
        magnification=mag_f,
        mag_source='printed' if mag_f is not None else 'none',
        mag_label=str(mag_label) if mag_label else None,
        region_xyxy=region_t,
        confidence=float(data.get('confidence') or (0.7 if nm_per_px else 0.2)),
        raw_text=raw_text,
        method_used='vlm',
        warnings=warnings,
    )
    return apply_scale_estimated_mag(reading)


# ---------------------------------------------------------------------------
# 통합 · 배치 · 배율 필터
# ---------------------------------------------------------------------------

def read_scale_bar(
    image_path: str | Path,
    *,
    method: ReadMethod = 'auto',
    client=None,
    y0_frac: float = INFO_BAND_Y0,
    known_label: tuple[float, str] | None = None,
) -> ScaleBarReading:
    """
    스케일바 읽기 엔트리.

    auto: CV → (nm_per_pixel 없으면) VLM
    """
    path = Path(image_path)
    if method == 'cv':
        return read_scale_bar_cv(path, y0_frac=y0_frac, known_label=known_label)
    if method == 'vlm':
        return read_scale_bar_vlm(path, client=client)

    # auto
    cv_r = read_scale_bar_cv(path, y0_frac=y0_frac, known_label=known_label)
    if cv_r.nm_per_pixel is not None and cv_r.found:
        return cv_r
    try:
        vlm_r = read_scale_bar_vlm(path, client=client)
    except Exception as e:  # noqa: BLE001
        cv_r.warnings.append(f'VLM 실패: {e}')
        return cv_r

    # --- CV 기하 + VLM 텍스트 병합 ---
    if cv_r.bar_px is not None:
        # 픽셀 폭은 CV가 더 안정적 → 우선 사용
        if vlm_r.bar_px is None or abs(vlm_r.bar_px - cv_r.bar_px) / cv_r.bar_px > 0.2:
            if vlm_r.bar_px is not None:
                vlm_r.warnings.append(
                    f'bar_px VLM({vlm_r.bar_px})→CV({cv_r.bar_px})로 교체'
                )
            vlm_r.bar_px = cv_r.bar_px
        if vlm_r.region_xyxy is None:
            vlm_r.region_xyxy = cv_r.region_xyxy

    if (
        vlm_r.label_value is not None
        and vlm_r.label_unit
        and vlm_r.bar_px
        and vlm_r.bar_px > 0
    ):
        phys = physical_nm(float(vlm_r.label_value), str(vlm_r.label_unit))
        if phys is not None:
            recomputed = phys / float(vlm_r.bar_px)
            if vlm_r.nm_per_pixel is None:
                vlm_r.nm_per_pixel = recomputed
            elif abs(float(vlm_r.nm_per_pixel) - recomputed) / max(recomputed, 1e-9) > 0.1:
                vlm_r.warnings.append(
                    f'nm_per_pixel VLM({vlm_r.nm_per_pixel:.4f})→재계산({recomputed:.4f})'
                )
                vlm_r.nm_per_pixel = recomputed
            vlm_r.found = True
            if vlm_r.kind == 'none':
                vlm_r.kind = cv_r.kind if cv_r.found else 'classic'

    if cv_r.found and not vlm_r.found:
        vlm_r.found = True
        vlm_r.kind = cv_r.kind
        vlm_r.bar_px = vlm_r.bar_px or cv_r.bar_px
        vlm_r.region_xyxy = vlm_r.region_xyxy or cv_r.region_xyxy
        vlm_r.warnings.append('VLM은 스케일 미확인, CV 막대 후보만 유지')

    vlm_r.method_used = 'auto(cv+vlm)' if cv_r.found else 'auto(vlm)'
    for w in cv_r.warnings:
        if w not in vlm_r.warnings:
            vlm_r.warnings.append(w)
    return vlm_r


def read_scale_bar_batch(
    image_paths: list[str | Path],
    *,
    method: ReadMethod = 'auto',
    client=None,
) -> ScaleBarBatchResult:
    readings = [
        read_scale_bar(p, method=method, client=client) for p in image_paths
    ]
    scale_map = {
        r.image_path: r.nm_per_pixel
        for r in readings
        if r.nm_per_pixel is not None and r.nm_per_pixel > 0
    }
    mag_map = {
        r.image_path: r.magnification
        for r in readings
        if r.magnification is not None and r.magnification > 0
    }
    return ScaleBarBatchResult(
        readings=readings,
        scale_map=scale_map,
        mag_map=mag_map,
    )


def filter_by_mag(
    readings: list[ScaleBarReading],
    *,
    mag_mode: Literal['max_only', 'all', 'custom'] = 'all',
    mag_custom: list[float] | None = None,
    rel_tol: float = 0.05,
) -> list[ScaleBarReading]:
    """
    기획서 mag_mode 분기.

    배율은 이미지에 숫자가 안 나온 경우가 많아, 각 reading에
    ``apply_scale_estimated_mag`` 로 스케일바 nm/px → 상대 배율을 채운 뒤 비교한다.

    - all: 그대로
    - max_only: 추정/표기 배율이 가장 큰 이미지만 (= nm/px 가 가장 작은 장)
    - custom: mag_custom 과 상대오차 rel_tol 이내
      (표기 절대 배율이 있는 장만 의미 있음. 스케일 추정값은 절대 kx와 스케일이 다름)
    """
    if mag_mode == 'all':
        return list(readings)

    filled = [apply_scale_estimated_mag(r.model_copy(deep=True)) for r in readings]
    with_mag = [r for r in filled if effective_magnification(r) is not None]
    if mag_mode == 'max_only':
        if not with_mag:
            return []
        mags = [effective_magnification(r) for r in with_mag]
        m = max(mags)  # type: ignore[type-var]
        return [
            r
            for r in with_mag
            if abs(effective_magnification(r) - m) / m <= rel_tol  # type: ignore[operator]
        ]

    # custom — 절대 배율 표기가 있는 장을 우선. 전부 추정이면 nm/px 역수로 느슨 매칭
    targets = mag_custom or []
    if not targets:
        return []
    out: list[ScaleBarReading] = []
    for r in with_mag:
        mag = effective_magnification(r)
        assert mag is not None
        # 추정 배율(mag_source=scale_estimated)은 절대 kx(50k)와 숫자 스케일이 다름
        if r.mag_source == 'scale_estimated':
            continue
        for t in targets:
            if t <= 0:
                continue
            if abs(mag - t) / t <= rel_tol:
                out.append(r)
                break
    return out


def filter_paths_by_mag_mode(
    image_paths: list[str],
    mag_map: dict[str, float],
    scale_map: dict[str, float] | None = None,
    *,
    mag_mode: Literal['max_only', 'all', 'custom'] = 'all',
    mag_custom: list[float] | None = None,
    rel_tol: float = 0.05,
) -> list[str]:
    """
    path 리스트만으로 mag_mode 필터 (State용 헬퍼).

    mag_map 이 비어 있고 scale_map 이 있으면 nm/px → 상대 배율로 채운다.
    """
    if mag_mode == 'all':
        return list(image_paths)

    eff: dict[str, float] = {}
    for p in image_paths:
        if p in mag_map and mag_map[p] > 0:
            eff[p] = float(mag_map[p])
        elif scale_map and p in scale_map and scale_map[p] > 0:
            eff[p] = estimate_mag_from_scale(scale_map[p])

    if mag_mode == 'max_only':
        if not eff:
            return []
        m = max(eff.values())
        return [p for p, v in eff.items() if abs(v - m) / m <= rel_tol]

    targets = mag_custom or []
    if not targets:
        return []
    out: list[str] = []
    for p, v in eff.items():
        # scale 추정만 있으면 custom 절대값 매칭 스킵
        if p not in mag_map and scale_map and p in scale_map:
            continue
        for t in targets:
            if t > 0 and abs(v - t) / t <= rel_tol:
                out.append(p)
                break
    return out


def draw_scale_overlay(
    image_path: str | Path,
    reading: ScaleBarReading,
    out_path: str | Path | None = None,
) -> np.ndarray:
    """검출 region / 막대를 시각화해 저장(선택)."""
    path = Path(image_path)
    bgr = _load_bgr(path)
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if reading.region_xyxy:
        x0, y0, x1, y1 = reading.region_xyxy
        p0 = (int(x0 * w), int(y0 * h))
        p1 = (int(x1 * w), int(y1 * h))
        cv2.rectangle(vis, p0, p1, (0, 255, 255), 2)
    label = f'{reading.kind}'
    if reading.nm_per_pixel:
        label += f' {reading.nm_per_pixel:.4f}nm/px'
    if reading.magnification:
        label += f' mag={reading.magnification:.0f}'
    cv2.putText(
        vis,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if out_path:
        _imwrite_unicode(Path(out_path), vis)
    return vis


# ---------------------------------------------------------------------------
# 합성 샘플 (테스트용)
# ---------------------------------------------------------------------------

def make_synthetic_scalebar_image(
    out_path: str | Path,
    *,
    width: int = 640,
    height: int = 480,
    bar_px: int = 120,
    label_value: float = 100.0,
    label_unit: str = 'nm',
    mag_kx: float = 50.0,
) -> Path:
    """하단 classic 스케일바가 있는 합성 SEM 스타일 이미지 저장."""
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    # 가짜 구조
    cv2.rectangle(img, (220, 120), (420, 360), (180, 180, 180), -1)
    # 정보바
    y0 = int(height * 0.88)
    img[y0:, :] = (20, 20, 20)
    # 스케일바 (왼쪽 하단)
    bx0, by0 = 24, y0 + 18
    cv2.rectangle(img, (bx0, by0), (bx0 + bar_px, by0 + 6), (240, 240, 240), -1)
    cv2.putText(
        img,
        f'{label_value:g} {label_unit}',
        (bx0, by0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f'{mag_kx:g}kx',
        (width - 120, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    _imwrite_unicode(outp, img)
    return outp


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    demo = root / 'data' / 'metrology' / 'synth_scalebar.png'
    make_synthetic_scalebar_image(demo, bar_px=120, label_value=100, label_unit='nm', mag_kx=50)
    r_cv = read_scale_bar_cv(demo, known_label=(100.0, 'nm'))
    print('CV synth:', r_cv.model_dump())
    tem = root / 'data' / 'metrology' / 'tem_sample_01.png'
    if tem.exists():
        r_tem = read_scale_bar_cv(tem)
        print('CV tem :', r_tem.model_dump())
