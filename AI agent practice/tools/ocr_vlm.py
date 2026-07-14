"""
원본 오버레이 OCR vs 측정 결과 이미지 OCR Tool.

기획서 ⑨a ``ocr_existing`` / ⑪ ``ocr_measured``
---------------------------------------------
입력이 다르다. 프롬프트·``source`` 태그도 분리한다.

| 노드 | 입력 이미지 | source | 역할 |
|------|-------------|--------|------|
| ocr_existing | 원본 (이미 측정선·숫자 있음) | ocr_existing | 1차 읽기 (truth) |
| ocr_measured | edge/measure가 **그린** 결과 PNG | ocr_measured | 검증용 (truth는 edge 수치) |

Usage
-----
    from tools.ocr_vlm import ocr_existing, ocr_measured

    read = ocr_existing(['sample_with_overlay.png'])
    verify = ocr_measured(['output/edge_overlay/sample_edge.png'])
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator

TableSource = Literal['ocr_existing', 'ocr_measured']
OcrMode = Literal['existing', 'measured']
Orientation = Literal['horizontal', 'vertical', 'angle']


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------

class Measurement(BaseModel):
    """VLM이 반환하는 측정값 1건."""

    value: float = Field(description='측정 숫자')
    unit: str = Field(description='단위 (nm, °, µm 등 이미지에 보이는 그대로)')
    anchor_x: float = Field(description='측정선이 닿는 구조 X (0~1)')
    anchor_y: float = Field(description='측정선이 닿는 구조 Y (0~1)')
    orientation: Orientation = Field(description='가로/세로/각도 측정')
    region: str = Field(
        default='',
        description='대략적 구역: top | middle | bottom | left | right',
    )
    goal_id: str = Field(
        default='',
        description='measure_goals 항목 id (매칭 실패 시 빈 문자열)',
    )

    @field_validator('anchor_x', 'anchor_y', mode='after')
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class ImageExtraction(BaseModel):
    measurements: list[Measurement] = Field(
        default_factory=list,
        description='이미지에서 찾은 측정값 목록',
    )


class OcrImageResult(BaseModel):
    """이미지 1장 OCR 결과."""

    image_path: str
    source: TableSource
    measurements: list[Measurement] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    ok: bool = True
    error: str = ''


class OcrBatchResult(BaseModel):
    """배치 OCR 결과."""

    source: TableSource
    results: list[OcrImageResult] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    n_images: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_measurements: int = 0


# ---------------------------------------------------------------------------
# 프롬프트 (입력 종류별로 분리)
# ---------------------------------------------------------------------------

_EXISTING_PROMPT = """이 SEM/TEM 단면 이미지에 **이미 그려져 있는** 측정 오버레이를 모두 읽으세요.

규칙:
1. 라벨 텍스트는 없을 수 있습니다. 숫자와 단위만 읽으세요.
2. 각 측정마다 녹색(또는 측정) 선이 닿는 구조 위치를
   anchor_x, anchor_y (이미지 왼쪽-위=0,0 ~ 오른쪽-아래=1,1)로 추정하세요.
3. 가로/세로/각도 측정을 orientation으로 표시하세요.
4. 숫자가 나란히 나열되어 있어도, 선이 가리키는 부위가 다르면 별도 항목입니다.
5. 읽기 순서로 묶지 마세요. 측정선이 닿는 부위 기준으로 구분하세요.
6. region은 top / middle / bottom / left / right 중 대략적으로 적어 주세요.
7. goal_id는 비워 두세요 (후단에서 매핑합니다).
"""

_MEASURED_PROMPT = """이 이미지는 Agent가 edge/measure로 **직접 그린** 측정 결과 오버레이입니다.
그려진 측정선·숫자·단위를 OCR로 읽어 검증용 목록을 만드세요.

규칙:
1. 원본 이미지에 원래 있던 스케일바/정보바 숫자는 무시하고,
   Agent가 그린 측정 오버레이(측정선·라벨)만 읽으세요.
2. 각 측정마다 선이 닿는 구조 위치를
   anchor_x, anchor_y (왼쪽-위=0,0 ~ 오른쪽-아래=1,1)로 추정하세요.
3. orientation은 horizontal / vertical / angle 중 하나로 표시하세요.
4. 숫자 나열 순서로 묶지 말고, 선이 가리키는 부위 기준으로 구분하세요.
5. region은 top / middle / bottom / left / right 중 대략적으로 적어 주세요.
6. goal_id는 비워 두세요.
7. 불확실하면 추측하지 말고 생략하세요 (검증용).
"""

_PROMPTS: dict[OcrMode, str] = {
    'existing': _EXISTING_PROMPT,
    'measured': _MEASURED_PROMPT,
}

_SOURCE_FOR_MODE: dict[OcrMode, TableSource] = {
    'existing': 'ocr_existing',
    'measured': 'ocr_measured',
}


# ---------------------------------------------------------------------------
# I/O · client
# ---------------------------------------------------------------------------

def encode_image(image_path: str | Path) -> tuple[str, str]:
    """로컬 이미지 → (mime_subtype, base64). mime_subtype 예: png, jpeg."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower().lstrip('.')
    mime = 'jpeg' if ext in ('jpg', 'jpeg') else (ext or 'png')
    b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
    return mime, b64


def _get_openai_client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY 가 없습니다 (.env)')
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# 핵심 추출
# ---------------------------------------------------------------------------

def extract_measurements(
    image_path: str | Path,
    *,
    mode: OcrMode = 'existing',
    model: str = 'gpt-4o',
    detail: str = 'high',
    client=None,
    measure_goals: list[str] | None = None,
) -> ImageExtraction:
    """
    이미지 1장 → ImageExtraction (structured).

    ``mode`` 에 따라 프롬프트가 달라진다.
    ``measure_goals`` 가 있으면 프롬프트에 목표 목록을 덧붙인다 (느슨한 힌트).
    """
    path = Path(image_path)
    mime, b64 = encode_image(path)
    prompt = _PROMPTS[mode]
    if measure_goals:
        goals = ', '.join(measure_goals)
        prompt = (
            f'{prompt}\n\n'
            f'가능하면 다음 측정 목표만 우선 추출하세요: [{goals}]\n'
            f'매칭되면 goal_id에 해당 id를 넣고, 불확실하면 비워 두세요.'
        )

    api = _get_openai_client(client)
    response = api.chat.completions.parse(
        model=model,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/{mime};base64,{b64}',
                        'detail': detail,
                    },
                },
            ],
        }],
        response_format=ImageExtraction,
        max_tokens=2000,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError('VLM이 구조화된 결과를 반환하지 않았습니다')
    return result


def measurements_to_rows(
    image_path: str | Path,
    measurements: list[Measurement] | list[dict[str, Any]],
    *,
    source: TableSource,
    magnification: float | None = None,
) -> list[dict[str, Any]]:
    """Measurement → Long row dict (goal_id · source · magnification 포함)."""
    from .table import measurements_to_rows as _to_long

    return _to_long(
        image_path,
        measurements,
        source=source,
        magnification=magnification,
    )

def ocr_one(
    image_path: str | Path,
    *,
    mode: OcrMode,
    model: str = 'gpt-4o',
    detail: str = 'high',
    client=None,
    measure_goals: list[str] | None = None,
    magnification: float | None = None,
) -> OcrImageResult:
    """이미지 1장 OCR (existing | measured)."""
    path = Path(image_path)
    source = _SOURCE_FOR_MODE[mode]
    try:
        extraction = extract_measurements(
            path,
            mode=mode,
            model=model,
            detail=detail,
            client=client,
            measure_goals=measure_goals,
        )
        rows = measurements_to_rows(
            path,
            extraction.measurements,
            source=source,
            magnification=magnification,
        )
        return OcrImageResult(
            image_path=str(path),
            source=source,
            measurements=extraction.measurements,
            rows=rows,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001 — 배치에서 실패 장만 건너뜀
        return OcrImageResult(
            image_path=str(path),
            source=source,
            ok=False,
            error=str(exc),
        )


def ocr_batch(
    image_paths: Iterable[str | Path],
    *,
    mode: OcrMode,
    model: str = 'gpt-4o',
    detail: str = 'high',
    client=None,
    measure_goals: list[str] | None = None,
    goal_filter_policy: Literal['drop', 'mark'] = 'drop',
    apply_goal_filter: bool = True,
    mag_map: dict[str, float] | None = None,
) -> OcrBatchResult:
    """여러 장 OCR. ``mode`` 에 따라 source 태그가 고정된다.

    ``measure_goals`` 가 있고 ``apply_goal_filter=True`` 이면
    추출 후 ``filter_by_goals`` 로 테이블 행을 제한한다.
    ``mag_map`` 이 있으면 각 행에 magnification 을 채운다.
    """
    from .recipe import filter_by_goals, parse_goals, summarize_goal_filter
    from .table import enrich_long_rows

    source = _SOURCE_FOR_MODE[mode]
    results: list[OcrImageResult] = []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    n_ok = 0
    n_failed = 0

    parsed = parse_goals(measure_goals=measure_goals or [])
    goals = parsed.measure_goals
    warnings.extend(parsed.warnings)
    mag_map = mag_map or {}

    def _mag_for(path: Path) -> float | None:
        s = str(path)
        if s in mag_map:
            return mag_map[s]
        if path.name in mag_map:
            return mag_map[path.name]
        try:
            rs = str(path.resolve())
            if rs in mag_map:
                return mag_map[rs]
        except OSError:
            pass
        return None

    paths = [Path(p) for p in image_paths]
    for path in paths:
        one = ocr_one(
            path,
            mode=mode,
            model=model,
            detail=detail,
            client=client,
            measure_goals=goals or None,
            magnification=_mag_for(path),
        )
        results.append(one)
        if one.ok:
            n_ok += 1
            rows.extend(one.rows)
        else:
            n_failed += 1
            warnings.append(f'{mode} OCR 실패: {path.name}: {one.error}')

    if apply_goal_filter and goals:
        fr = filter_by_goals(rows, goals, policy=goal_filter_policy)
        warnings.extend(fr.warnings)
        warnings.append(f'goal_filter: {summarize_goal_filter(fr)}')
        rows = fr.kept

    if mag_map:
        rows = enrich_long_rows(rows, mag_map=mag_map)

    return OcrBatchResult(
        source=source,
        results=results,
        rows=rows,
        warnings=warnings,
        n_images=len(paths),
        n_ok=n_ok,
        n_failed=n_failed,
        n_measurements=len(rows),
    )


def ocr_existing(
    image_paths: Iterable[str | Path],
    *,
    model: str = 'gpt-4o',
    detail: str = 'high',
    client=None,
    measure_goals: list[str] | None = None,
    goal_filter_policy: Literal['drop', 'mark'] = 'drop',
    apply_goal_filter: bool = True,
    mag_map: dict[str, float] | None = None,
) -> OcrBatchResult:
    """⑨a: 원본 측정 오버레이 읽기 → source=ocr_existing."""
    return ocr_batch(
        image_paths,
        mode='existing',
        model=model,
        detail=detail,
        client=client,
        measure_goals=measure_goals,
        goal_filter_policy=goal_filter_policy,
        apply_goal_filter=apply_goal_filter,
        mag_map=mag_map,
    )


def ocr_measured(
    image_paths: Iterable[str | Path],
    *,
    model: str = 'gpt-4o',
    detail: str = 'high',
    client=None,
    measure_goals: list[str] | None = None,
    goal_filter_policy: Literal['drop', 'mark'] = 'drop',
    apply_goal_filter: bool = True,
    mag_map: dict[str, float] | None = None,
) -> OcrBatchResult:
    """⑪: measure가 그린 결과 이미지 OCR (검증) → source=ocr_measured."""
    return ocr_batch(
        image_paths,
        mode='measured',
        model=model,
        detail=detail,
        client=client,
        measure_goals=measure_goals,
        goal_filter_policy=goal_filter_policy,
        apply_goal_filter=apply_goal_filter,
        mag_map=mag_map,
    )


def summarize_ocr_batch(batch: OcrBatchResult) -> str:
    return (
        f"ocr source={batch.source} images={batch.n_images} "
        f"ok={batch.n_ok} fail={batch.n_failed} "
        f"measurements={batch.n_measurements}"
    )


# ---------------------------------------------------------------------------
# LangGraph 노드 헬퍼 (State dict / Pydantic 모두 수용)
# ---------------------------------------------------------------------------

def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _paths_from_state(
    state: Any,
    *,
    preferred_keys: tuple[str, ...],
    fallback_keys: tuple[str, ...] = (),
) -> list[str]:
    for key in preferred_keys + fallback_keys:
        val = _state_get(state, key)
        if val:
            return [str(p) for p in val]
    single = _state_get(state, 'image_path')
    if single:
        return [str(single)]
    return []


def ocr_existing_node(state: Any) -> dict[str, Any]:
    """
    LangGraph 노드: 원본 오버레이 OCR.

    읽기 경로 우선순위: ``read_paths`` → ``image_paths`` → ``image_path``
    결과는 ``raw_measurements`` 에 source=ocr_existing 로 기록.
    ``measure_goals`` / ``goal_text`` 가 있으면 filter_by_goals 적용.
    ``mag_map`` 이 있으면 magnification 컬럼 채움.
    """
    from .recipe import goals_from_state

    paths = _paths_from_state(
        state,
        preferred_keys=('read_paths',),
        fallback_keys=('image_paths',),
    )
    goals = goals_from_state(state)
    mag_map = dict(_state_get(state, 'mag_map') or {})

    if not paths:
        return {
            'raw_measurements': [],
            'warnings': ['ocr_existing: 읽을 이미지 경로 없음'],
        }

    batch = ocr_existing(
        paths,
        measure_goals=goals or None,
        mag_map=mag_map or None,
    )
    print(f'[ocr_existing] {summarize_ocr_batch(batch)}')
    prev_warn = list(_state_get(state, 'warnings') or [])
    return {
        'raw_measurements': batch.rows,
        'measure_goals': goals,
        'warnings': prev_warn + batch.warnings,
    }


def ocr_measured_node(state: Any) -> dict[str, Any]:
    """
    LangGraph 노드: 측정 결과 이미지 OCR (검증).

    경로 우선순위: ``measured_image_paths`` → ``verify_paths``
    결과는 ``ocr_verify_rows`` 에 source=ocr_measured 로 기록
    (1차 truth는 edge ``raw_measurements`` 를 유지).
    """
    from .recipe import goals_from_state

    paths = _paths_from_state(
        state,
        preferred_keys=('measured_image_paths', 'verify_paths'),
    )
    goals = goals_from_state(state)
    mag_map = dict(_state_get(state, 'mag_map') or {})

    if not paths:
        return {
            'ocr_verify_rows': [],
            'warnings': ['ocr_measured: 검증할 measured 이미지 경로 없음'],
        }

    batch = ocr_measured(
        paths,
        measure_goals=goals or None,
        mag_map=mag_map or None,
    )
    print(f'[ocr_measured] {summarize_ocr_batch(batch)}')
    prev_warn = list(_state_get(state, 'warnings') or [])
    return {
        'ocr_verify_rows': batch.rows,
        'measure_goals': goals,
        'warnings': prev_warn + batch.warnings,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    sample = root / 'data' / 'metrology' / 'tem_sample_01.png'
    if not sample.exists():
        print('sample missing:', sample)
    else:
        print('module OK; sample exists:', sample.name)
        print('prompts:', list(_PROMPTS))
        print('node helpers: ocr_existing_node / ocr_measured_node')
