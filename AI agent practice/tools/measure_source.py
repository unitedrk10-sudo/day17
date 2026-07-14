"""
읽기(OCR existing) vs edge 생성 측정 분기 Tool.

기획서 ⑧ 이후 · source 컬럼
----------------------------
``classify_overlay_batch`` 의 삼분 결과(``OverlayBranchResult``)를 받아
이미지마다 처리 행동과 Long 테이블 ``source`` 태그를 확정한다.

| overlay_mode | 있는 장 | 없는 장 |
|--------------|---------|---------|
| all          | ocr_existing (읽기) | — |
| some         | ocr_existing (읽기 + recipe 교사) | edge_measure (생성) |
| none         | — | edge_measure (생성) |

``source`` 값
- ``ocr_existing`` : 원본에 있던 측정값을 읽음
- ``edge``         : edge Tool이 수치를 생성 (권장 truth)
- ``ocr_measured`` : edge가 그린 결과 이미지를 다시 OCR (검증용, 옵션)

Usage
-----
    from tools.overlay import classify_overlay_batch
    from tools.measure_source import plan_measure_sources

    branch = classify_overlay_batch(paths)
    plan = plan_measure_sources(branch, verify_edge_with_ocr=False)

    for p in plan.read_paths:
        ...  # ocr_existing
    for p in plan.edge_paths:
        ...  # edge_measure  → source='edge'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .overlay import OverlayBranchResult, OverlayMode, RouteName, route_overlay_mode

MeasureAction = Literal['ocr_existing', 'edge_measure']
TableSource = Literal['ocr_existing', 'edge', 'ocr_measured']


class ImageSourcePlan(BaseModel):
    """이미지 1장에 대한 읽기/생성 계획."""

    image_path: str
    action: MeasureAction = Field(
        description='실제로 돌릴 파이프: 원본 OCR vs edge 측정',
    )
    source: TableSource = Field(
        description='Long 테이블에 넣을 1차 source 태그',
    )
    has_overlay: bool = False
    use_as_recipe_teacher: bool = Field(
        default=False,
        description='some 모드에서 어노테이션 장 → recipe 학습 소스',
    )
    run_ocr_verify: bool = Field(
        default=False,
        description='edge 후 그린 이미지에 ocr_measured 검증 실행 여부',
    )


class MeasureSourcePlan(BaseModel):
    """배치 단위 읽기/생성 분기 결과."""

    overlay_mode: OverlayMode
    overlay_route: RouteName
    plans: list[ImageSourcePlan] = Field(default_factory=list)
    read_paths: list[str] = Field(
        default_factory=list,
        description='action=ocr_existing',
    )
    edge_paths: list[str] = Field(
        default_factory=list,
        description='action=edge_measure',
    )
    recipe_teacher_paths: list[str] = Field(
        default_factory=list,
        description='some일 때 recipe 학습에 쓸 오버레이 장',
    )
    verify_paths: list[str] = Field(
        default_factory=list,
        description='edge 후 ocr_measured 검증 대상',
    )
    verify_edge_with_ocr: bool = False

    @property
    def needs_read(self) -> bool:
        return bool(self.read_paths)

    @property
    def needs_edge(self) -> bool:
        return bool(self.edge_paths)

    @property
    def needs_recipe_learn(self) -> bool:
        return bool(self.recipe_teacher_paths) and bool(self.edge_paths)


def plan_measure_sources(
    branch: OverlayBranchResult,
    *,
    verify_edge_with_ocr: bool = False,
) -> MeasureSourcePlan:
    """
    삼분 분기 결과 → 이미지별 읽기/edge 생성 계획.

    Parameters
    ----------
    branch :
        ``classify_overlay_batch`` 반환값 (mode + with/without 경로).
    verify_edge_with_ocr :
        True면 edge로 그린 장에 대해 ``run_ocr_verify`` 도 켬
        (테이블 1차 truth는 여전히 ``edge``, OCR은 검증).
    """
    mode = branch.mode
    overlay_route = route_overlay_mode(branch)
    plans: list[ImageSourcePlan] = []

    with_set = set(branch.with_overlay)
    without_set = set(branch.without_overlay)

    # results 순서를 유지해 plan 생성 (실패 장도 without에 포함됨)
    ordered_paths: list[str] = []
    seen: set[str] = set()
    for r in branch.results:
        if r.image_path not in seen:
            ordered_paths.append(r.image_path)
            seen.add(r.image_path)
    for p in list(branch.with_overlay) + list(branch.without_overlay):
        if p not in seen:
            ordered_paths.append(p)
            seen.add(p)

    for path in ordered_paths:
        has = path in with_set
        if mode == 'all':
            plans.append(
                ImageSourcePlan(
                    image_path=path,
                    action='ocr_existing',
                    source='ocr_existing',
                    has_overlay=True,
                    use_as_recipe_teacher=False,
                    run_ocr_verify=False,
                )
            )
        elif mode == 'none':
            plans.append(
                ImageSourcePlan(
                    image_path=path,
                    action='edge_measure',
                    source='edge',
                    has_overlay=False,
                    use_as_recipe_teacher=False,
                    run_ocr_verify=verify_edge_with_ocr,
                )
            )
        else:  # some
            if has:
                plans.append(
                    ImageSourcePlan(
                        image_path=path,
                        action='ocr_existing',
                        source='ocr_existing',
                        has_overlay=True,
                        use_as_recipe_teacher=True,
                        run_ocr_verify=False,
                    )
                )
            else:
                plans.append(
                    ImageSourcePlan(
                        image_path=path,
                        action='edge_measure',
                        source='edge',
                        has_overlay=False,
                        use_as_recipe_teacher=False,
                        run_ocr_verify=verify_edge_with_ocr,
                    )
                )

    read_paths = [p.image_path for p in plans if p.action == 'ocr_existing']
    edge_paths = [p.image_path for p in plans if p.action == 'edge_measure']
    teachers = [p.image_path for p in plans if p.use_as_recipe_teacher]
    verify = [p.image_path for p in plans if p.run_ocr_verify]

    return MeasureSourcePlan(
        overlay_mode=mode,
        overlay_route=overlay_route,
        plans=plans,
        read_paths=read_paths,
        edge_paths=edge_paths,
        recipe_teacher_paths=teachers,
        verify_paths=verify,
        verify_edge_with_ocr=verify_edge_with_ocr,
    )


def plan_from_mode_paths(
    mode: OverlayMode,
    *,
    with_overlay: Iterable[str | Path] = (),
    without_overlay: Iterable[str | Path] = (),
    verify_edge_with_ocr: bool = False,
) -> MeasureSourcePlan:
    """
    OverlayBranchResult 없이 mode + 경로 리스트만으로 계획 생성.
    (이미 삼분 결과의 with/without 만 State에 있을 때)
    """
    with_list = [str(Path(p)) for p in with_overlay]
    without_list = [str(Path(p)) for p in without_overlay]
    # 가짜 branch — results는 경로만 채움
    from .overlay import OverlayImageResult

    results = [
        OverlayImageResult(image_path=p, has_overlay=True, score=1.0)
        for p in with_list
    ] + [
        OverlayImageResult(image_path=p, has_overlay=False, score=0.0)
        for p in without_list
    ]
    branch = OverlayBranchResult(
        mode=mode,
        results=results,
        with_overlay=with_list,
        without_overlay=without_list,
        n_total=len(results),
        n_with=len(with_list),
        n_without=len(without_list),
    )
    return plan_measure_sources(branch, verify_edge_with_ocr=verify_edge_with_ocr)


def tag_rows(
    rows: list[dict[str, Any]],
    *,
    source: TableSource,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    """측정 row dict 들에 source 컬럼을 붙이거나 덮어쓴다."""
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if image_path is not None:
            r.setdefault('image_path', image_path)
        r['source'] = source
        out.append(r)
    return out


def merge_read_and_edge_rows(
    *,
    read_rows: list[dict[str, Any]] | None = None,
    edge_rows: list[dict[str, Any]] | None = None,
    ocr_measured_rows: list[dict[str, Any]] | None = None,
    mag_map: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    읽기·edge·검증 OCR 결과를 한 Long 리스트로 합친다.
    각각에 source 가 없으면 기본 태그를 부여하고,
    mag_map 이 있으면 magnification 을 채운다.
    """
    from .table import enrich_long_rows

    merged: list[dict[str, Any]] = []
    for row in read_rows or []:
        r = dict(row)
        r.setdefault('source', 'ocr_existing')
        merged.append(r)
    for row in edge_rows or []:
        r = dict(row)
        r.setdefault('source', 'edge')
        merged.append(r)
    for row in ocr_measured_rows or []:
        r = dict(row)
        r.setdefault('source', 'ocr_measured')
        merged.append(r)
    return enrich_long_rows(merged, mag_map=mag_map or None)


def summarize_measure_source_plan(plan: MeasureSourcePlan) -> str:
    return (
        f"source_plan mode={plan.overlay_mode} route={plan.overlay_route} "
        f"read={len(plan.read_paths)} edge={len(plan.edge_paths)} "
        f"teachers={len(plan.recipe_teacher_paths)} "
        f"verify_ocr={len(plan.verify_paths)}"
    )


def route_after_source_plan(plan: MeasureSourcePlan) -> str:
    """
    LangGraph 상위 라우트 (한 번에 하나만 갈 때).

    - read만 → ocr_existing
    - edge만 → edge_measure
    - 둘 다 → parallel_read_and_edge (또는 learn_and_measure)
    """
    if plan.needs_read and plan.needs_edge:
        return 'parallel_read_and_edge'
    if plan.needs_read:
        return 'ocr_existing'
    if plan.needs_edge:
        return 'edge_measure'
    return 'export'


if __name__ == '__main__':
    from .overlay import classify_overlay_batch

    root = Path(__file__).resolve().parents[1]
    metro = root / 'data' / 'metrology'
    paths = [
        p
        for p in [
            metro / 'tem_sample_01.png',
            metro / 'synth_scalebar.png',
            metro / 'uploads' / 'ref_pat.png',
        ]
        if p.exists()
    ]
    branch = classify_overlay_batch(paths)
    plan = plan_measure_sources(branch, verify_edge_with_ocr=True)
    print(summarize_measure_source_plan(plan))
    print('route_after', route_after_source_plan(plan))
    for p in plan.plans:
        print(
            f"  {Path(p.image_path).name:24s} action={p.action:14s} "
            f"source={p.source:14s} teacher={p.use_as_recipe_teacher} "
            f"verify={p.run_ocr_verify}"
        )
