"""
parse_request — mag_mode + measure_goals 입력 바인딩 · State 초기화.

기획서 §2 · §4 ① · §14.3 (오케스트레이션 노드)
-------------------------------------------------
START 직후 호출. UI/API/노트북에서 받은 옵션을 정규화해 State에 고정한다.

- ``mag_mode``: max_only | all | custom
  (배율 숫자는 이미지에 없는 경우가 많음 → 이후 ``read_scale_bar``가
  스케일바 nm/px로 상대 배율을 추정하고, 이 정책으로 필터)
- ``measure_goals`` / ``goal_text``: ``parse_goals`` 로 표준 id 정규화
- 둘 다 비면 goals=[] (= 전부 측정)

Usage
-----
    from tools.parse_request import parse_request, parse_request_node, RequestOptions

    result = parse_request(
        input_uri='data/metrology',
        mag_mode='max_only',
        goal_text='기둥 상단 폭과 높이만. 각도 제외',
    )
    # result.state → LangGraph invoke 입력
"""

from __future__ import annotations

import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, field_validator

from .recipe import STANDARD_GOAL_IDS, edge_kinds_from_goals, parse_goals

MagMode = Literal['max_only', 'all', 'custom']


class RequestOptions(BaseModel):
    """실행 시점 옵션 (기획서 RequestOptions)."""

    mag_mode: MagMode = 'all'
    mag_custom: list[float] = Field(default_factory=list)
    measure_goals: list[str] = Field(default_factory=list)
    goal_text: str = ''


class ParseRequestInput(BaseModel):
    """UI / API / 노트북에서 받는 원본 입력."""

    input_uri: str = ''
    mag_mode: MagMode = 'all'
    mag_custom: list[float] = Field(default_factory=list)
    # custom 입력: "50k, 100000, 80k" 문자열도 허용 (API 편의)
    mag_custom_text: str = ''
    measure_goals: list[str] = Field(default_factory=list)
    goal_text: str = ''

    @field_validator('mag_mode', mode='before')
    @classmethod
    def _norm_mag_mode(cls, v: Any) -> str:
        s = str(v or 'all').strip().lower()
        aliases = {
            'max': 'max_only',
            'maximum': 'max_only',
            '최고': 'max_only',
            '최대': 'max_only',
            '전체': 'all',
            '지정': 'custom',
            '선택': 'custom',
        }
        return aliases.get(s, s)


class ParseRequestResult(BaseModel):
    """parse_request 정규화 결과."""

    ok: bool = True
    input_uri: str = ''
    options: RequestOptions = Field(default_factory=RequestOptions)
    # LangGraph START 입력용 평면 State 패치
    state: dict[str, Any] = Field(default_factory=dict)
    edge_kinds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    excluded_goals: list[str] = Field(default_factory=list)
    summary: str = ''


def parse_mag_custom_text(text: str) -> list[float]:
    """
    ``\"50k, 100000, 80k\"`` → ``[50000.0, 100000.0, 80000.0]``.
    숫자·k/K 접미사만 지원.
    """
    if not (text or '').strip():
        return []
    out: list[float] = []
    for part in re.split(r'[,;\s]+', text.strip()):
        if not part:
            continue
        m = re.fullmatch(r'([\d.]+)\s*([kK])?', part)
        if not m:
            continue
        val = float(m.group(1))
        if m.group(2):
            val *= 1000.0
        out.append(val)
    return out


def parse_request(
    *,
    input_uri: str = '',
    mag_mode: MagMode | str = 'all',
    mag_custom: Sequence[float] | None = None,
    mag_custom_text: str = '',
    measure_goals: Sequence[str] | None = None,
    goal_text: str = '',
) -> ParseRequestResult:
    """
    옵션 파싱 · State 초기화 패치 생성.

    Returns
    -------
    ParseRequestResult
        ``.state`` 를 ``graph.invoke(result.state)`` 에 넘기면 된다.
    """
    raw = ParseRequestInput(
        input_uri=input_uri or '',
        mag_mode=mag_mode,  # type: ignore[arg-type]
        mag_custom=list(mag_custom or []),
        mag_custom_text=mag_custom_text or '',
        measure_goals=list(measure_goals or []),
        goal_text=goal_text or '',
    )
    warnings: list[str] = []

    mode = raw.mag_mode
    if mode not in ('max_only', 'all', 'custom'):
        warnings.append(f'알 수 없는 mag_mode={mode!r} → all')
        mode = 'all'

    mags = list(raw.mag_custom)
    if raw.mag_custom_text.strip():
        parsed_mags = parse_mag_custom_text(raw.mag_custom_text)
        if not parsed_mags and raw.mag_custom_text.strip():
            warnings.append(f'mag_custom_text 파싱 실패: {raw.mag_custom_text!r}')
        for m in parsed_mags:
            if m not in mags:
                mags.append(m)

    if mode == 'custom' and not mags:
        warnings.append('mag_mode=custom 인데 mag_custom 비어 있음 → all 로 폴백')
        mode = 'all'

    goals_parsed = parse_goals(
        measure_goals=raw.measure_goals,
        goal_text=raw.goal_text,
    )
    warnings.extend(goals_parsed.warnings)

    options = RequestOptions(
        mag_mode=mode,  # type: ignore[arg-type]
        mag_custom=mags,
        measure_goals=list(goals_parsed.measure_goals),
        goal_text=goals_parsed.goal_text or raw.goal_text,
    )

    kinds = sorted(edge_kinds_from_goals(options.measure_goals or None))
    uri = raw.input_uri.strip()
    if not uri:
        warnings.append('input_uri 비어 있음 (필수 권장)')

    state: dict[str, Any] = {
        'input_uri': uri,
        'options': options.model_dump(),
        'mag_mode': options.mag_mode,
        'mag_custom': list(options.mag_custom),
        'measure_goals': list(options.measure_goals),
        'goal_text': options.goal_text,
        'warnings': list(warnings),
        # 이후 노드용 빈 슬롯
        'image_paths': [],
        'scale_map': {},
        'mag_map': {},
        'overlay_mode': '',
        'recipe': None,
        'measured_image_paths': [],
        'raw_measurements': [],
        'ocr_verify_rows': [],
        'table_rows': [],
        'summary': '',
    }

    goals_label = ', '.join(options.measure_goals) if options.measure_goals else '(all)'
    mag_label = options.mag_mode
    if options.mag_mode == 'custom':
        mag_label = f'custom{options.mag_custom}'
    summary = (
        f"uri={uri or '(empty)'} · mag={mag_label} · goals=[{goals_label}]"
    )

    return ParseRequestResult(
        ok=True,
        input_uri=uri,
        options=options,
        state=state,
        edge_kinds=kinds,
        warnings=warnings,
        excluded_goals=list(goals_parsed.excluded),
        summary=summary,
    )


def parse_request_node(state: Any) -> dict[str, Any]:
    """
    LangGraph 노드: State에 있는 입력 필드를 정규화해 덮어쓴다.

    기대 입력 키: ``input_uri``, ``mag_mode``, ``mag_custom``,
    ``mag_custom_text``, ``measure_goals``, ``goal_text`` (또는 ``options``).
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    opts = _get(state, 'options')
    mag_mode = _get(state, 'mag_mode')
    mag_custom = _get(state, 'mag_custom')
    measure_goals = _get(state, 'measure_goals')
    goal_text = _get(state, 'goal_text')

    if opts is not None:
        if mag_mode is None:
            mag_mode = _get(opts, 'mag_mode', 'all')
        if not mag_custom:
            mag_custom = _get(opts, 'mag_custom') or []
        if not measure_goals:
            measure_goals = _get(opts, 'measure_goals') or []
        if not goal_text:
            goal_text = _get(opts, 'goal_text') or ''

    result = parse_request(
        input_uri=str(_get(state, 'input_uri') or ''),
        mag_mode=mag_mode or 'all',
        mag_custom=list(mag_custom or []),
        mag_custom_text=str(_get(state, 'mag_custom_text') or ''),
        measure_goals=list(measure_goals or []),
        goal_text=str(goal_text or ''),
    )
    # 기존 warnings 유지 + 신규 병합
    prev_warn = list(_get(state, 'warnings') or [])
    merged = list(dict.fromkeys(prev_warn + result.warnings))
    out = dict(result.state)
    out['warnings'] = merged
    out['summary'] = result.summary
    print(f'[parse_request] {result.summary}')
    if merged:
        for w in merged:
            print(f'  ! {w}')
    return out


def list_goal_catalog() -> list[dict[str, Any]]:
    """UI 체크박스용 표준 goal 목록."""
    from .recipe import GOAL_SPECS

    rows = []
    for gid in STANDARD_GOAL_IDS:
        spec = GOAL_SPECS[gid]
        rows.append(
            {
                'goal_id': gid,
                'orientation': spec.orientation,
                'edge_kinds': list(spec.edge_kinds),
                'aliases': list(spec.aliases[:6]),
            }
        )
    return rows


def summarize_parse(result: ParseRequestResult) -> str:
    lines = [result.summary]
    if result.edge_kinds:
        lines.append(f'edge_kinds={result.edge_kinds}')
    if result.excluded_goals:
        lines.append(f'excluded={result.excluded_goals}')
    if result.warnings:
        lines.append(f'warnings={len(result.warnings)}')
    return ' | '.join(lines)
