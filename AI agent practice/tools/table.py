"""
Long 테이블 변환 · 컬럼 정규화 Tool.

기획서 Long 컬럼
----------------
``image_path``, ``image_name``, ``magnification``, ``slot_key``,
``goal_id``, ``value``, ``unit``, ``anchor_x``, ``anchor_y``,
``orientation``, ``region``, ``source``
(``source`` ∈ ocr_existing | ocr_measured | edge)

Usage
-----
    from tools.table import measurements_to_rows, rows_to_dataframe, enrich_long_rows

    rows = measurements_to_rows(path, measurements, source='ocr_existing', magnification=50000)
    df = rows_to_dataframe(rows)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pandas as pd

TableSource = Literal['ocr_existing', 'ocr_measured', 'edge']

# 기획서 §6 컬럼 순서
LONG_COLUMNS: tuple[str, ...] = (
    'image_path',
    'image_name',
    'magnification',
    'slot_key',
    'goal_id',
    'value',
    'unit',
    'anchor_x',
    'anchor_y',
    'orientation',
    'region',
    'source',
)

ORIENT_PREFIX = {
    'horizontal': 'h',
    'vertical': 'v',
    'angle': 'a',
}


def _quantize(coord: float, step: float = 0.05) -> float:
    return round(round(float(coord) / step) * step, 2)


def assign_slot_key(
    m: Any,
    *,
    step: float = 0.05,
    include_goal: bool = False,
) -> str:
    """anchor + orientation → slot_key (예: v_0.50_0.20). include_goal이면 접미사."""
    if hasattr(m, 'model_dump'):
        data = m.model_dump()
    elif isinstance(m, dict):
        data = m
    else:
        raise TypeError(type(m))

    orient = str(data.get('orientation') or 'x')
    prefix = ORIENT_PREFIX.get(orient, 'x')
    qx = _quantize(float(data.get('anchor_x', 0.0)), step)
    qy = _quantize(float(data.get('anchor_y', 0.0)), step)
    key = f'{prefix}_{qx:.2f}_{qy:.2f}'
    if include_goal:
        gid = str(data.get('goal_id') or '').strip()
        if gid and gid != 'unmatched_goal':
            key = f'{key}__{gid}'
    return key


def _as_dict(m: Any) -> dict[str, Any]:
    if hasattr(m, 'model_dump'):
        return m.model_dump()
    if isinstance(m, dict):
        return dict(m)
    raise TypeError(f'dict 또는 BaseModel 필요: {type(m)}')


def normalize_long_row(
    row: dict[str, Any],
    *,
    image_path: str | Path | None = None,
    source: str | None = None,
    magnification: float | None = None,
    slot_step: float = 0.05,
    default_source: str = '',
) -> dict[str, Any]:
    """한 행을 Long 스키마로 정규화 (누락 컬럼 채움)."""
    r = dict(row)
    path: Path | None = None
    if image_path is not None:
        path = Path(image_path)
    elif r.get('image_path'):
        path = Path(str(r['image_path']))

    if path is not None:
        r['image_path'] = str(path.resolve()) if path.exists() else str(path)
        r.setdefault('image_name', path.name)
    else:
        r.setdefault('image_path', '')
        r.setdefault('image_name', '')

    if magnification is not None:
        r['magnification'] = magnification
    else:
        r.setdefault('magnification', None)

    if source is not None:
        r['source'] = source
    else:
        r.setdefault('source', default_source)

    r.setdefault('goal_id', '')
    r.setdefault('region', '')
    r.setdefault('unit', '')

    if 'slot_key' not in r or not r['slot_key']:
        r['slot_key'] = assign_slot_key(r, step=slot_step)

    # 숫자 필드
    for key in ('value', 'anchor_x', 'anchor_y'):
        if key in r and r[key] is not None:
            r[key] = float(r[key])
    if r.get('magnification') is not None and r['magnification'] != '':
        try:
            r['magnification'] = float(r['magnification'])
        except (TypeError, ValueError):
            r['magnification'] = None

    # 스키마 순서 + 여분 키 유지(여분은 뒤에)
    ordered = {c: r.get(c) for c in LONG_COLUMNS}
    for k, v in r.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def measurements_to_rows(
    image_path: str | Path,
    measurements: Sequence[Any],
    *,
    source: TableSource | str = 'ocr_existing',
    magnification: float | None = None,
    slot_step: float = 0.05,
    goal_id_default: str = '',
) -> list[dict[str, Any]]:
    """Measurement 리스트 → Long 테이블 행들."""
    rows: list[dict[str, Any]] = []
    path = Path(image_path)
    for m in measurements:
        data = _as_dict(m)
        raw = {
            'value': data.get('value'),
            'unit': data.get('unit', ''),
            'anchor_x': data.get('anchor_x'),
            'anchor_y': data.get('anchor_y'),
            'orientation': data.get('orientation', ''),
            'region': data.get('region', ''),
            'goal_id': data.get('goal_id') or goal_id_default,
            'source': data.get('source') or source,
        }
        rows.append(
            normalize_long_row(
                raw,
                image_path=path,
                source=str(raw['source']),
                magnification=magnification,
                slot_step=slot_step,
            )
        )
    return rows


def enrich_long_rows(
    rows: Iterable[dict[str, Any]],
    *,
    mag_map: dict[str, float] | None = None,
    scale_map: dict[str, float] | None = None,  # noqa: ARG001 — 예약(nm/px)
    source: str | None = None,
    default_magnification: float | None = None,
) -> list[dict[str, Any]]:
    """
    기존 row에 magnification / source / goal_id 를 보강.

    mag_map: image_path → magnification
    """
    mag_map = mag_map or {}
    # path / name 양쪽 키 허용
    mag_by_name = {Path(k).name: v for k, v in mag_map.items()}

    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        path_s = str(r.get('image_path') or '')
        name = str(r.get('image_name') or Path(path_s).name)
        mag = r.get('magnification')
        if mag is None or mag == '':
            mag = mag_map.get(path_s)
            if mag is None and path_s:
                # resolve 키 매칭 시도
                for k, v in mag_map.items():
                    try:
                        if Path(k).resolve() == Path(path_s).resolve():
                            mag = v
                            break
                    except OSError:
                        continue
            if mag is None:
                mag = mag_by_name.get(name, default_magnification)
        if source is not None and not r.get('source'):
            r['source'] = source
        out.append(
            normalize_long_row(
                r,
                magnification=float(mag) if mag is not None else r.get('magnification'),
            )
        )
    return out


def rows_to_dataframe(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Long rows → DataFrame (기획 컬럼 순서)."""
    if not rows:
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    normalized = [normalize_long_row(r) for r in rows]
    df = pd.DataFrame(normalized)
    # 필수 컬럼 보장
    for c in LONG_COLUMNS:
        if c not in df.columns:
            df[c] = None
    extra = [c for c in df.columns if c not in LONG_COLUMNS]
    return df[list(LONG_COLUMNS) + extra]


def merge_long_tables(
    *frames: pd.DataFrame | list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """여러 Long 소스 병합."""
    parts: list[pd.DataFrame] = []
    for f in frames:
        if f is None:
            continue
        if isinstance(f, pd.DataFrame):
            if len(f):
                parts.append(rows_to_dataframe(f.to_dict(orient='records')))
        elif isinstance(f, list) and f:
            parts.append(rows_to_dataframe(f))
    if not parts:
        return rows_to_dataframe([])
    return pd.concat(parts, ignore_index=True)


def save_long_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    encoding: str = 'utf-8-sig',
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_to_dataframe(df.to_dict(orient='records') if len(df) else []).to_csv(
        out, index=False, encoding=encoding
    )
    return out


if __name__ == '__main__':
    demo = measurements_to_rows(
        'demo.png',
        [
            {
                'value': 34.3,
                'unit': 'nm',
                'anchor_x': 0.5,
                'anchor_y': 0.2,
                'orientation': 'vertical',
                'region': 'height',
                'goal_id': 'height',
            }
        ],
        source='edge',
        magnification=50000.0,
    )
    df = rows_to_dataframe(demo)
    print(df.columns.tolist())
    print(df.to_string(index=False))
