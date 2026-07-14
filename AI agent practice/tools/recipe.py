"""
measure_goals → recipe / edge kinds / OCR·테이블 필터.

기획서 §2.2 · §14 ``filter_by_goals`` / ``parse_goals`` / ``learn_recipe``
----------------------------------------------------------------------
목표가 비어 있으면 **전부** 측정·유지한다.
목표가 있으면:

1. ``parse_goals``        — 자연어/별칭 → 표준 goal id
2. ``edge_kinds_from_goals`` — edge Tool이 잴 region base 집합
3. ``filter_by_goals``    — Measurement/row에 ``goal_id`` 부여 후 drop|mark
4. ``learn_recipe``       — 교사 측정에서 목표 부위로 패턴 레시피 학습

Usage
-----
    from tools.recipe import parse_goals, filter_by_goals, edge_kinds_from_goals

    goals = parse_goals(goal_text='기둥 상단 폭과 높이만. 각도 제외')
    kinds = edge_kinds_from_goals(goals)  # {'height', 'top'}
    kept, dropped = filter_by_goals(rows, goals, policy='drop')
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 표준 goal id (기획서 미결 항목 → 여기서 확정)
# ---------------------------------------------------------------------------

# edge `_measure_one_mask` region base 이름과 1:1에 가깝게 맞춤
EdgeKind = Literal['height', 'top', 'middle', 'bottom', 'right_sidewall', 'left_sidewall']
FilterPolicy = Literal['drop', 'mark']

STANDARD_GOAL_IDS: tuple[str, ...] = (
    'height',
    'top_width',
    'middle_width',
    'bottom_width',
    'right_sidewall_angle',
    'left_sidewall_angle',
)


class GoalSpec(BaseModel):
    """표준 goal 1개 메타."""

    goal_id: str
    edge_kinds: list[str] = Field(description='sem_edge_tool region base')
    orientation: Literal['horizontal', 'vertical', 'angle'] | None = None
    # OCR/row 매칭용 키워드 (region·텍스트)
    region_keys: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


GOAL_SPECS: dict[str, GoalSpec] = {
    'height': GoalSpec(
        goal_id='height',
        edge_kinds=['height'],
        orientation='vertical',
        region_keys=['height', 'ht'],
        aliases=['height', 'h', '높이', '두께', 'thickness'],
    ),
    'top_width': GoalSpec(
        goal_id='top_width',
        edge_kinds=['top'],
        orientation='horizontal',
        region_keys=['top', 'top_width', 'cd_top'],
        aliases=[
            'top_width', 'top', 'topwidth', 'cd_top',
            '상단', '상단폭', '상단 폭', '꼭대기', '머리',
        ],
    ),
    'middle_width': GoalSpec(
        goal_id='middle_width',
        edge_kinds=['middle'],
        orientation='horizontal',
        region_keys=['middle', 'mid', 'middle_width'],
        aliases=[
            'middle_width', 'mid_width', 'middle', 'mid',
            '중간', '중간폭', '중폭',
        ],
    ),
    'bottom_width': GoalSpec(
        goal_id='bottom_width',
        edge_kinds=['bottom'],
        orientation='horizontal',
        region_keys=['bottom', 'bot', 'bottom_width'],
        aliases=[
            'bottom_width', 'bot_width', 'bottom', 'bot',
            '하단', '하단폭', '바닥', '발',
        ],
    ),
    'right_sidewall_angle': GoalSpec(
        goal_id='right_sidewall_angle',
        edge_kinds=['right_sidewall'],
        orientation='angle',
        region_keys=['right_sidewall', 'sidewall', 'rsw'],
        aliases=[
            'right_sidewall_angle', 'right_sidewall', 'sidewall',
            'sidewall_angle', 'angle', 'rsw',
            '측벽', '측벽각', '각도', '오른쪽 각도', '우측각',
        ],
    ),
    'left_sidewall_angle': GoalSpec(
        goal_id='left_sidewall_angle',
        edge_kinds=['left_sidewall'],
        orientation='angle',
        region_keys=['left_sidewall', 'lsw'],
        aliases=[
            'left_sidewall_angle', 'left_sidewall', 'lsw',
            '왼쪽 각도', '좌측각', '좌측벽',
        ],
    ),
}


# region base → 표준 goal
_KIND_TO_GOAL: dict[str, str] = {}
for _gid, _spec in GOAL_SPECS.items():
    for _k in _spec.edge_kinds:
        _KIND_TO_GOAL[_k] = _gid


class GoalFilterResult(BaseModel):
    """filter_by_goals 결과."""

    kept: list[dict[str, Any]] = Field(default_factory=list)
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    measure_goals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    n_matched: int = 0
    n_unmatched: int = 0


class ParsedGoals(BaseModel):
    measure_goals: list[str] = Field(default_factory=list)
    goal_text: str = ''
    warnings: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 정규화 · 파싱
# ---------------------------------------------------------------------------

def _norm_token(s: str) -> str:
    t = (s or '').strip().lower()
    t = t.replace('-', '_').replace(' ', '_')
    t = re.sub(r'_+', '_', t)
    return t


def normalize_goal_id(raw: str) -> str | None:
    """별칭/표기 → 표준 goal_id. 모르면 None."""
    token = _norm_token(raw)
    if not token:
        return None
    if token in GOAL_SPECS:
        return token
    # µm 등 장식 제거
    token2 = re.sub(r'(_?nm|_?um|_?µm|_?px|_?deg|_?degree|°)$', '', token)
    if token2 in GOAL_SPECS:
        return token2
    for gid, spec in GOAL_SPECS.items():
        for alias in spec.aliases:
            if _norm_token(alias) == token or _norm_token(alias) == token2:
                return gid
    return None


def region_base(region: str) -> str:
    """``p0_top`` / ``top`` → ``top``."""
    r = (region or '').strip().lower()
    if not r:
        return ''
    # p12_right_sidewall → right_sidewall
    m = re.match(r'^p\d+_(.+)$', r)
    if m:
        return m.group(1)
    if '_' in r and r.split('_', 1)[0].startswith('p') and r[1:2].isdigit():
        return r.split('_', 1)[-1]
    return r


def goal_id_from_region(region: str) -> str | None:
    base = region_base(region)
    if not base:
        return None
    if base in _KIND_TO_GOAL:
        return _KIND_TO_GOAL[base]
    # top_width 형태로 region이 온 경우
    return normalize_goal_id(base)


def edge_kinds_from_goals(measure_goals: Sequence[str] | None) -> set[str]:
    """
    goals → edge가 계산할 region base 집합.
    goals 비어 있으면 전체 kinds.
    """
    if not measure_goals:
        return set(_KIND_TO_GOAL.keys())
    kinds: set[str] = set()
    for g in measure_goals:
        gid = normalize_goal_id(g) or (g if g in GOAL_SPECS else None)
        if gid and gid in GOAL_SPECS:
            kinds.update(GOAL_SPECS[gid].edge_kinds)
    return kinds


def parse_goals(
    *,
    measure_goals: Sequence[str] | None = None,
    goal_text: str = '',
) -> ParsedGoals:
    """
    구조화 리스트 + 자연어 → 표준 ``measure_goals``.

    - 리스트가 있으면 정규화해 사용
    - ``goal_text`` 키워드로 보강/제외 (예: "각도 제외")
    - 둘 다 비면 ``[]`` (= 전부)
    """
    warnings: list[str] = []
    excluded: list[str] = []
    found: list[str] = []

    for raw in measure_goals or []:
        gid = normalize_goal_id(str(raw))
        if gid:
            if gid not in found:
                found.append(gid)
        else:
            warnings.append(f'알 수 없는 goal 무시: {raw!r}')

    text = (goal_text or '').strip()
    text_l = text.lower()

    if text:
        # 제외 패턴 (키워드 매칭보다 먼저)
        exclude_angle = bool(
            re.search(r'각도\s*제외|angle\s*除外|no\s*angle|각도\s*빼', text, re.I)
            or re.search(r'except\s+angle|without\s+angle', text_l)
        )
        exclude_width = bool(re.search(r'폭\s*제외|width\s*제외|no\s*width', text, re.I))
        exclude_height = bool(re.search(r'높이\s*제외|height\s*제외|no\s*height', text, re.I))

        # 제외 구문 제거한 텍스트로 포함 키워드 검색
        text_for_include = text
        if exclude_angle:
            text_for_include = re.sub(
                r'각도\s*제외|no\s*angle|except\s+angle|without\s+angle|각도\s*빼',
                ' ',
                text_for_include,
                flags=re.I,
            )
        if exclude_width:
            text_for_include = re.sub(
                r'폭\s*제외|no\s*width|width\s*제외', ' ', text_for_include, flags=re.I
            )
        if exclude_height:
            text_for_include = re.sub(
                r'높이\s*제외|no\s*height|height\s*제외', ' ', text_for_include, flags=re.I
            )

        # 포함 키워드 (텍스트에서 직접)
        keyword_map: list[tuple[str, list[str]]] = [
            ('height', [r'높이', r'height', r'두께']),
            ('top_width', [r'상단\s*폭', r'상단', r'top\s*width', r'\btop\b', r'꼭대기']),
            ('middle_width', [r'중간\s*폭', r'중폭', r'middle\s*width', r'mid\s*width']),
            ('bottom_width', [r'하단\s*폭', r'바닥', r'bottom\s*width', r'bot\s*width']),
            ('right_sidewall_angle', [r'측벽', r'측벽각', r'각도', r'sidewall', r'\bangle\b']),
            ('left_sidewall_angle', [r'왼쪽\s*각', r'좌측각', r'left\s*sidewall']),
        ]
        text_hits: list[str] = []
        for gid, pats in keyword_map:
            for pat in pats:
                if re.search(pat, text_for_include, re.I):
                    text_hits.append(gid)
                    break

        # "폭과 높이"처럼 폭만 포괄 언급 → top_width 기본 (전부 폭은 loose)
        if re.search(r'폭', text_for_include) and not any(
            g.endswith('_width') for g in text_hits
        ):
            text_hits.append('top_width')

        if text_hits and not found:
            for g in text_hits:
                if g not in found:
                    found.append(g)
        elif text_hits and found:
            # 리스트 + 텍스트: 합집합
            for g in text_hits:
                if g not in found:
                    found.append(g)

        if exclude_angle:
            for g in ('right_sidewall_angle', 'left_sidewall_angle'):
                if g in found:
                    found.remove(g)
                    excluded.append(g)
            # 텍스트만 있고 아직 비어 있으면 각도를 뺀 나머지 전부
            if not found and not measure_goals:
                found = [g for g in STANDARD_GOAL_IDS if g not in excluded]

        if exclude_width:
            for g in list(found):
                if g.endswith('_width'):
                    found.remove(g)
                    excluded.append(g)

        if exclude_height and 'height' in found:
            found.remove('height')
            excluded.append('height')

        if not found and not measure_goals and not (
            exclude_angle or exclude_width or exclude_height
        ):
            warnings.append(
                f'goal_text 정규화 실패 → 전부 측정: {goal_text!r}'
            )

    return ParsedGoals(
        measure_goals=found,
        goal_text=text,
        warnings=warnings,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# 매칭 · 필터
# ---------------------------------------------------------------------------

def _as_row(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, 'model_dump'):
        return item.model_dump()
    raise TypeError(f'dict 또는 BaseModel 필요: {type(item)}')


def match_goal_id(
    item: Any,
    measure_goals: Sequence[str] | None = None,
) -> str | None:
    """
    Measurement/row → 가장 잘 맞는 표준 goal_id.
    ``measure_goals`` 가 있으면 그 안에서만 고른다.
    """
    row = _as_row(item)
    allowed = None
    if measure_goals:
        allowed = {
            normalize_goal_id(g) or g
            for g in measure_goals
            if (normalize_goal_id(g) or g) in GOAL_SPECS
        }
        if not allowed:
            allowed = None

    # 1) 이미 goal_id
    existing = normalize_goal_id(str(row.get('goal_id') or ''))
    if existing and (allowed is None or existing in allowed):
        return existing

    # 2) region → kind → goal
    from_region = goal_id_from_region(str(row.get('region') or ''))
    if from_region and (allowed is None or from_region in allowed):
        return from_region

    # 3) orientation + region_keys 휴리스틱 (단어 경계로 부분일치)
    orient = str(row.get('orientation') or '').lower()
    region_l = str(row.get('region') or '').lower()
    unit_l = str(row.get('unit') or '').lower()

    candidates: list[tuple[int, str]] = []  # (score, goal_id)
    for gid, spec in GOAL_SPECS.items():
        if allowed is not None and gid not in allowed:
            continue
        score = 0
        if spec.orientation and orient == spec.orientation:
            score += 2
        for key in spec.region_keys:
            if not key:
                continue
            # 짧은 키(<=2)는 토큰 완전일치만
            if len(key) <= 2:
                tokens = re.split(r'[^a-z0-9가-힣]+', region_l)
                if key in tokens:
                    score += 3
                    break
            elif key in region_l:
                score += 3
                break
        # 각도 단위
        if gid.endswith('_angle') and ('°' in (row.get('unit') or '') or 'deg' in unit_l):
            score += 2
        if score > 0:
            candidates.append((score, gid))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def assign_goal_ids(
    items: Iterable[Any],
    measure_goals: Sequence[str] | None = None,
    *,
    unmatched_label: str = 'unmatched_goal',
) -> list[dict[str, Any]]:
    """모든 항목에 ``goal_id`` 부여 (실패 시 unmatched_label)."""
    out: list[dict[str, Any]] = []
    for item in items:
        row = _as_row(item)
        gid = match_goal_id(row, measure_goals)
        row['goal_id'] = gid if gid else unmatched_label
        out.append(row)
    return out


def filter_by_goals(
    items: Iterable[Any],
    measure_goals: Sequence[str] | None = None,
    *,
    policy: FilterPolicy = 'drop',
    unmatched_label: str = 'unmatched_goal',
) -> GoalFilterResult:
    """
    goals에 맞는 측정만 남긴다.

    - goals 비어 있으면 전부 유지 (goal_id는 region에서 추정, 없으면 '')
    - ``drop``: 미매칭 제거
    - ``mark``: 미매칭도 남기되 goal_id=unmatched_goal
    """
    goals_list = [
        normalize_goal_id(g) or g
        for g in (measure_goals or [])
        if (normalize_goal_id(g) or g) in GOAL_SPECS
    ]
    warnings: list[str] = []

    if not goals_list:
        kept: list[dict[str, Any]] = []
        for item in items:
            row = _as_row(item)
            if not row.get('goal_id'):
                inferred = match_goal_id(row, None)
                row['goal_id'] = inferred or ''
            kept.append(row)
        return GoalFilterResult(
            kept=kept,
            dropped=[],
            measure_goals=[],
            warnings=warnings,
            n_matched=len(kept),
            n_unmatched=0,
        )

    kept = []
    dropped = []
    n_matched = 0
    n_unmatched = 0

    for item in items:
        row = _as_row(item)
        gid = match_goal_id(row, goals_list)
        if gid and gid in goals_list:
            row['goal_id'] = gid
            kept.append(row)
            n_matched += 1
        else:
            n_unmatched += 1
            row['goal_id'] = unmatched_label
            if policy == 'mark':
                kept.append(row)
            else:
                dropped.append(row)

    if n_matched == 0 and n_unmatched > 0:
        warnings.append(
            'goals 필터 매칭 0건 — 입력이 애매하면 policy=mark 또는 goals 비우기 검토'
        )

    return GoalFilterResult(
        kept=kept,
        dropped=dropped,
        measure_goals=goals_list,
        warnings=warnings,
        n_matched=n_matched,
        n_unmatched=n_unmatched,
    )


def summarize_goal_filter(result: GoalFilterResult) -> str:
    return (
        f"goals={result.measure_goals or ['(all)']} "
        f"kept={len(result.kept)} dropped={len(result.dropped)} "
        f"matched={result.n_matched} unmatched={result.n_unmatched}"
    )


# ---------------------------------------------------------------------------
# recipe 학습 (goals로 교사 측정 제한)
# ---------------------------------------------------------------------------

def learn_recipe(
    measurements: Sequence[Any],
    measure_goals: Sequence[str] | None = None,
    *,
    multi_pattern: bool | None = None,
    match_threshold: float = 0.15,
):
    """
    어노테이션/측정 리스트 → ``MeasureRecipe`` (+ enabled_goals/kinds).

    goals가 있으면 해당 측정만 앵커로 쓰고, recipe에 enabled_* 를 기록한다.
    """
    # 지연 import: sem_edge_tool ↔ tools 순환 방지
    from sem_edge_tool import Measurement as EdgeMeasurement
    from sem_edge_tool import MeasureRecipe, build_recipe_from_measurements

    filt = filter_by_goals(measurements, measure_goals, policy='drop')
    use_rows = filt.kept if filt.measure_goals else [_as_row(m) for m in measurements]
    if filt.measure_goals and not use_rows:
        # 목표 매칭 실패 시 전체로 fallback (기획: 느슨)
        use_rows = [_as_row(m) for m in measurements]
        filt.warnings.append('learn_recipe: goals 매칭 실패 → 전체 측정으로 recipe 학습')

    edge_ms = [
        EdgeMeasurement(
            value=float(r['value']),
            unit=str(r.get('unit') or 'px'),
            anchor_x=float(r['anchor_x']),
            anchor_y=float(r['anchor_y']),
            orientation=r.get('orientation') or 'vertical',
            region=str(r.get('region') or ''),
            goal_id=str(r.get('goal_id') or ''),
        )
        for r in use_rows
        if 'value' in r and 'anchor_x' in r and 'anchor_y' in r
    ]
    recipe = build_recipe_from_measurements(
        edge_ms,
        multi_pattern=multi_pattern,
        match_threshold=match_threshold,
    )
    kinds = sorted(edge_kinds_from_goals(filt.measure_goals or None))
    goals = list(filt.measure_goals)
    # MeasureRecipe에 enabled_* 필드가 있으면 채움
    fields = getattr(recipe, 'model_fields', {}) or {}
    updates: dict[str, Any] = {}
    if 'enabled_goals' in fields:
        updates['enabled_goals'] = goals
    if 'enabled_kinds' in fields:
        updates['enabled_kinds'] = kinds
    if updates:
        recipe = recipe.model_copy(update=updates)
    return recipe, filt


def learn_recipe_node(state: Any) -> dict[str, Any]:
    """
    LangGraph ⑨b: 교사 측정(raw 또는 recipe_teacher OCR) → recipe.

    State 키: ``raw_measurements`` 또는 ``teacher_measurements``,
    ``measure_goals`` / ``goal_text``.
    """
    goals = goals_from_state(state)
    teachers = (
        _get_list(state, 'teacher_measurements')
        or _get_list(state, 'raw_measurements')
        or []
    )
    if not teachers:
        return {
            'recipe': None,
            'warnings': ['learn_recipe: 교사 측정 없음'],
            'measure_goals': goals,
        }
    recipe, filt = learn_recipe(teachers, goals or None)
    prev = list(_get_list(state, 'warnings') or [])
    print(
        f'[learn_recipe] patterns={len(recipe.patterns)} '
        f'goals={recipe.enabled_goals or goals} '
        f'kinds={recipe.enabled_kinds}'
    )
    return {
        'recipe': recipe.model_dump(),
        'measure_goals': goals or list(filt.measure_goals),
        'warnings': prev + filt.warnings,
    }


def edge_measure_with_goals(
    image_paths: Sequence[str | Path],
    measure_goals: Sequence[str] | None = None,
    *,
    recipe: Any = None,
    nm_per_pixel: float | None = None,
    save_overlay: bool = True,
    mag_map: dict[str, float] | None = None,
    **kwargs: Any,
):
    """
    goals를 ``measure_sem_batch`` / ``measure_sem_structure`` 에 전달하는 래퍼.
    Long rows에 source=edge · goal_id · magnification 포함.
    """
    from sem_edge_tool import MeasureRecipe, measure_sem_batch, measure_sem_structure

    from .table import enrich_long_rows, normalize_long_row

    goals = list(measure_goals or [])
    kinds = edge_kinds_from_goals(goals) if goals else None
    recipe_obj = recipe
    if isinstance(recipe, dict):
        recipe_obj = MeasureRecipe(**recipe)

    paths = [Path(p) for p in image_paths]
    rows: list[dict[str, Any]] = []
    measured_paths: list[str] = []
    active_recipe = recipe_obj
    mag_map = mag_map or {}

    if len(paths) == 0:
        return [], None, []

    if recipe_obj is None and len(paths) > 1:
        results, active_recipe = measure_sem_batch(
            paths,
            nm_per_pixel=nm_per_pixel,
            save_overlay=save_overlay,
            measure_goals=goals or None,
            measure_kinds=kinds,
            **kwargs,
        )
    else:
        results = []
        for i, p in enumerate(paths):
            ext = measure_sem_structure(
                p,
                nm_per_pixel=nm_per_pixel,
                save_overlay=save_overlay,
                recipe=active_recipe if i > 0 or recipe_obj is not None else None,
                measure_goals=goals or None,
                measure_kinds=kinds,
                **kwargs,
            )
            results.append(ext)
            if active_recipe is None and ext.recipe is not None:
                active_recipe = ext.recipe

    def _mag_for(p: Path) -> float | None:
        s = str(p)
        if s in mag_map:
            return mag_map[s]
        if p.name in mag_map:
            return mag_map[p.name]
        try:
            rs = str(p.resolve())
            if rs in mag_map:
                return mag_map[rs]
        except OSError:
            pass
        return None

    for p, ext in zip(paths, results):
        mag = _mag_for(p)
        for m in ext.measurements:
            r = m.model_dump() if hasattr(m, 'model_dump') else dict(m)
            r['source'] = 'edge'
            if not r.get('goal_id'):
                r['goal_id'] = goal_id_from_region(str(r.get('region') or '')) or ''
            rows.append(
                normalize_long_row(
                    r,
                    image_path=p,
                    source='edge',
                    magnification=mag,
                )
            )
        if save_overlay:
            overlay = (
                Path(kwargs.get('overlay_dir') or 'data/metrology/output/edge_overlay')
                / f'{p.stem}_edge.png'
            )
            if overlay.exists():
                measured_paths.append(str(overlay.resolve()))

    if goals:
        fr = filter_by_goals(rows, goals, policy='drop')
        rows = fr.kept

    if mag_map:
        rows = enrich_long_rows(rows, mag_map=mag_map, source='edge')

    return rows, active_recipe, measured_paths

def _get_list(state: Any, key: str) -> list[Any]:
    if isinstance(state, dict):
        val = state.get(key)
    else:
        val = getattr(state, key, None)
    return list(val) if val else []


def goals_from_state(state: Any) -> list[str]:
    """State / dict / options 에서 measure_goals 추출 (+ goal_text 파싱)."""
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    goals = list(_get(state, 'measure_goals') or [])
    goal_text = str(_get(state, 'goal_text') or '')
    opts = _get(state, 'options')
    if opts is not None:
        if not goals:
            goals = list(_get(opts, 'measure_goals') or [])
        if not goal_text:
            goal_text = str(_get(opts, 'goal_text') or '')

    parsed = parse_goals(measure_goals=goals, goal_text=goal_text)
    return parsed.measure_goals


def edge_measure_node(state: Any) -> dict[str, Any]:
    """LangGraph: goals 반영 edge 측정 → raw_measurements + measured_image_paths."""
    goals = goals_from_state(state)
    paths = (
        _get_list(state, 'edge_paths')
        or _get_list(state, 'image_paths')
        or ([_get(state, 'image_path')] if _get(state, 'image_path') else [])
    )
    paths = [p for p in paths if p]
    recipe = _get(state, 'recipe')
    scale_map = _get(state, 'scale_map') or {}
    mag_map = dict(_get(state, 'mag_map') or {})
    nm = None
    if scale_map and paths:
        nm = scale_map.get(str(paths[0])) or next(iter(scale_map.values()), None)

    rows, recipe_out, measured = edge_measure_with_goals(
        paths,
        goals or None,
        recipe=recipe,
        nm_per_pixel=float(nm) if nm else None,
        mag_map=mag_map or None,
    )
    prev = _get_list(state, 'warnings')
    print(f'[edge_measure] goals={goals or "(all)"} rows={len(rows)}')
    out: dict[str, Any] = {
        'raw_measurements': rows,
        'measured_image_paths': measured,
        'measure_goals': goals,
        'warnings': prev,
    }
    if recipe_out is not None:
        out['recipe'] = (
            recipe_out.model_dump()
            if hasattr(recipe_out, 'model_dump')
            else recipe_out
        )
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



if __name__ == "__main__":
    samples = [
        {
            "value": 10, "unit": "nm", "anchor_x": 0.5, "anchor_y": 0.2,
            "orientation": "vertical", "region": "p0_height",
        },
        {
            "value": 5, "unit": "nm", "anchor_x": 0.5, "anchor_y": 0.15,
            "orientation": "horizontal", "region": "p0_top",
        },
        {
            "value": 88, "unit": "deg", "anchor_x": 0.6, "anchor_y": 0.4,
            "orientation": "angle", "region": "p0_right_sidewall",
        },
    ]
    g = parse_goals(goal_text="top width and height only, no angle")
    # Korean example also works via aliases in parse_goals
    g2 = parse_goals(goal_text="기둥 상단 폭과 높이만. 각도 제외")
    print("parsed EN:", g.measure_goals)
    print("parsed KO:", g2.measure_goals)
    print("kinds:", edge_kinds_from_goals(g2.measure_goals))
    fr = filter_by_goals(samples, g2.measure_goals, policy="drop")
    print(summarize_goal_filter(fr))
    for r in fr.kept:
        print(" ", r["region"], "->", r["goal_id"])
