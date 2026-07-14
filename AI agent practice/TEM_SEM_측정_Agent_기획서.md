# TEM/SEM 측정 Agent 기획서 v2

> **문서 역할:** 구현과 분리된 기획 원본. 구현은 `AIagent_practice.ipynb` + `sem_edge_tool.py`.  
> **환경:** Conda `base` · OpenAI VLM (`gpt-4o`) · LangGraph · pandas · OpenCV  
> **참고 실습:** 14.2 VLM · 15.1~15.4 LangGraph · 16.2 Sub-Graph  
> **주석 규칙:** `<!-- YYYY-MM-DD | 추가|수정|분리 | 한 줄 설명 -->` — 해당 블록이 **언제·왜** 들어왔는지 표시. 본문 흐름은 최신(v2) 기준.

---

## 1. 한 줄 정의

사용자가 **입력(파일/폴더/URL/zip) + 옵션 프롬프트**를 주면, TEM/SEM 단면 이미지를 수집·필터하고,  
**이미 찍힌 측정값은 OCR/VLM으로**, **없는 경우엔 학습·엣지 측정으로** 보강한 뒤,  
앵커 기준 Long 테이블로 누적한다.

---

## 2. 입력 (파일 + 프롬프트)

실행 시점에 **경로/URI와 함께** 아래를 선택·작성한다. LangGraph `START` 직후 `parse_request` 노드에서 State에 고정한다.


| 필드              | 타입        | 설명                     | 기본                 | 비고          |
| --------------- | --------- | ---------------------- | ------------------ | ----------- |
| `input_uri`     | str       | 로컬 경로 · 폴더 · URL · zip | (필수)               |             |
| `mag_mode`      | enum      | 배율 필터 정책               | `all`              |             |
| `measure_goals` | list[str] | 자연어                    | **어디를 / 무엇을** 측정할지 | `[]` (= 전부) |
| `goal_text`     | str       | 자연어 원문 (로그·프롬프트용)      | `""`               |             |




### 2.1 배율 모드 (`mag_mode`) — 초기에 path 분기

**전제:** 운영 이미지에는 `50.0kx` 같은 **배율 숫자가 없는 경우가 많다**.  
배율은 스케일바(`nm/px`)로 **상대 추정**한다 (`tools.scale_bar.estimate_mag_from_scale`).  
`nm/px`가 작을수록 배율이 크다. `max_only`는 이 추정값으로 “가장 큰 배율” 장을 고른다.


| 값          | 동작                                         |
| ---------- | ------------------------------------------ |
| `max_only` | 스케일바에서 추정한 배율 중 **최대만** 측정 (nm/px 최소)     |
| `all`      | 배율 필터 없이 **수집된 이미지 전부** 측정                 |
| `custom`   | (고급) 정보바에 절대 배율 표기가 **있을 때만** 지정값과 매칭     |


```
parse_request
    ├─ mag_mode == max_only  →  … → read_scale_bar → filter_mag (추정 최대만)
    └─ mag_mode == all       →  … → (filter_mag 스킵, 전량 유지)
```

배율 필터 **정책**은 입력 옵션이다. 절대 배율 숫자는 사용자가 모르는 것이 정상이며, UI는 all / max_only를 기본으로 쓴다.

### 2.2 측정 목표 (`measure_goals`)

이미 “어디를 재라”는 목표가 있을 수 있다. 없으면 이미지에 보이는(또는 엣지로 잡는) **전부**를 대상으로 한다.

**예시 입력**

- 구조화: `["top_width", "height", "right_sidewall_angle"]`
- 자연어: `"기둥 상단 폭과 높이만. 각도 제외"`

**반영 지점 (전 구간 공통)**


| 단계               | `measure_goals` 효과                                |
| ---------------- | ------------------------------------------------- |
| 오버레이 검출·OCR      | 목표에 해당하는 측정만 테이블에 남김 (나머지는 drop 또는 `ignored`)     |
| 레시피 학습 (6-2)     | **목표 부위만** recipe 슬롯으로 학습                         |
| 엣지 측정 (6-3 / 7)  | `sem_edge_tool`에 넘길 region / 측정 종류를 목표로 제한        |
| slot / region 명명 | 가능하면 goal id ↔ `region` / `slot_key` 매핑           |
| 출력               | Long 테이블에 `goal_id` 컬럼 (매칭 실패 시 `unmatched_goal`) |


자연어는 `parse_request`에서 VLM/규칙으로 `measure_goals` 리스트로 정규화한다.  
매핑 실패 시 보수적으로 **전부 측정 후** goal 필터를 느슨하게 적용하고, 로그에 경고를 남긴다.

---



## 3. 배경 · 대상

- TEM / SEM **단면** 이미지
- 실제 운영: 라벨 없이 `34.3nm`, `77.9°` 등 **숫자+단위만** 있는 경우 많음
- 오버레이가 **전부 / 일부 / 없음** 세 경우 모두 지원
- 압축(zip) · URL · 로컬 파일/폴더 입력

---



## 4. 전체 시퀀스 (LangGraph)

```
[입력] input_uri + mag_mode + measure_goals
    │
    ▼
① parse_request     — 옵션 파싱 · State 초기화          <!-- 07-14 -->
    │
    ▼
② resolve_input     — 링크면 다운로드 / 로컬이면 경로 확정  <!-- 07-13 -->
    │
    ▼
③ classify_archive  — zip이면 ④ unzip, 아니면 ⑤로       <!-- 07-13 -->
    │
    ▼
④ unzip (조건부)                                         <!-- 07-13 -->
    │
    ▼
⑤ collect_images    — 이미지 파일 리스트                 <!-- 07-13 -->
    │
    ▼
⑥ read_scale_bar    — nm/px · 배율 메타 → State.scale_map <!-- 07-14 추가: 필터 전에 확정 -->
    │                  ※ 배율 필터보다 먼저 (환산·필터 공통 전제)
    │
    ▼
⑦ filter_by_mag     — mag_mode에 따라 분기               <!-- 07-14 -->
    │   · max_only → 최대 배율만
    │   · all      → 스킵(전량)
    │   · custom   → 지정 배율만
    │
    ▼
⑧ detect_overlay    — 이미지별 “측정 오버레이 유무” 집계  <!-- 07-13 이분 → 07-14 삼분 -->
    │
    ├─ all_have   → ⑨a ocr_existing                       <!-- 07-14: OCR 노드 분리 -->
    ├─ some_have  → ⑨b learn_from_annotated → ⑩ measure → ⑪ ocr_measured
    │                 (어노테이션 없는 장만 measure; 있는 장은 OCR)
    └─ none_have  → ⑨c edge_measure (또는 goals 기반 measure) <!-- 07-13 edge -->
                         → ⑩ save overlays → ⑪ ocr_measured
    │
    ▼
⑫ slot_and_table    — 앵커·goal 기준 slot_key · Long 누적 <!-- 07-10 앵커 / 07-14 goal -->
    │
    ▼
⑬ export            — DataFrame / CSV                     <!-- 07-10 · 07-13 CSV -->
    │
    ▼
   END
```



### 4.1 노드별 메모


| #     | 노드               | LangGraph  | 비고                            |
| ----- | ---------------- | ---------- | ----------------------------- |
| 1     | `parse_request`  | 필수         | `mag_mode`, `measure_goals`   |
| 2–5   | intake           | 서브그래프 가능   | zip/URL은 순수 함수·**Tool**로도 충분  |
| 6     | `read_scale_bar` | 필수         | 실패 시 px 단위 + 경고               |
| 7     | `filter_by_mag`  | **조건부 엣지** | 사용자 선택 path                   |
| 8     | `detect_overlay` | **조건부 엣지** | all / some / none             |
| 9a    | `ocr_existing`   |            | 원본 오버레이 읽기                    |
| 9b    | `learn_recipe`   |            | 어노테이션 1장+로 MeasureRecipe      |
| 9c–10 | `measure`        |            | edge + recipe + goals         |
| 11    | `ocr_measured`   |            | **측정 결과 이미지** OCR (9a와 입력 다름) |
| 12–13 | table / export   |            |                               |


---



## 5. 오버레이 분기 상세 (⑧)


| 케이스      | 조건                    | 경로                                                     |
| -------- | --------------------- | ------------------------------------------------------ |
| **all**  | 대상 이미지 모두에 측정 오버레이 있음 | → `ocr_existing` → table (측정 스킵)                       |
| **some** | 일부만 있음                | 있는 장: OCR / 없는 장: 있는 장에서 **goals에 맞는 방식 학습** 후 measure |
| **none** | 없음                    | → `edge_measure` (+ goals) → 오버레이 저장 → `ocr_measured`  |


`some`에서 학습 소스가 여러 장이면: goals에 가장 잘 맞는 1장을 고르거나, 합집합 recipe를 만든다 (구현 Phase에서 결정).

---



## 6. 핵심 설계: 묶음 · 앵커 · 목표

**같은 묶음** = 측정선이 닿는 **구조 위 동일 부위** + **동일 방향** (+ 가능하면 동일 `goal_id`)

- 화면상 숫자 나열 순서 ≠ 같은 묶음
- `slot_key`: orientation + 정규화 앵커 (+ goal_id)
- 이미지 간: 앵커 거리 + orientation (+ goal)로 정렬, 실패 시 `unmatched`



### 스키마 (초안)

```python
class Measurement(BaseModel):
    value: float
    unit: str
    anchor_x: float          # 0~1
    anchor_y: float
    orientation: str         # horizontal | vertical | angle
    region: str = ""
    goal_id: str = ""        # measure_goals 항목 id, 없으면 ""  <!-- 07-14 -->

class RequestOptions(BaseModel):  # <!-- 07-14 추가 -->
    mag_mode: Literal["max_only", "all", "custom"] = "all"
    mag_custom: list[float] = []
    measure_goals: list[str] = []
    goal_text: str = ""

class MetrologyState(BaseModel):
    input_uri: str
    options: RequestOptions                    # <!-- 07-14 -->
    image_paths: list[str] = []                # <!-- 07-13 -->
    scale_map: dict[str, float] = {}           # <!-- 07-14 -->
    mag_map: dict[str, float] = {}             # <!-- 07-14 --> path → magnification
    overlay_mode: str = ""                     # all | some | none  <!-- 07-14 -->
    recipe: dict | None = None                 # <!-- 07-13 -->
    measured_image_paths: list[str] = []
    raw_measurements: list[dict] = []          # <!-- 07-10 --> · ocr_existing / edge 1차
    ocr_verify_rows: list[dict] = []           # <!-- 07-14 --> ocr_measured 검증용
    table_rows: list[dict] = []                # <!-- 07-10 -->
    warnings: list[str] = []
    summary: str = ""
```



### Long 테이블 컬럼

`image_path`, `image_name`, `magnification`, `slot_key`, `goal_id`, `value`, `unit`, `anchor_x`, `anchor_y`, `orientation`, `region`, `source` (`ocr_existing` | `ocr_measured` | `edge`)

---



## 7. LangGraph 조립 스케치

```
START → parse_request → resolve_input → classify_archive
      → (unzip?) → collect_images → read_scale_bar
      → filter_by_mag          # mag_mode 조건부
      → detect_overlay
           ├─ all  → ocr_existing ────────────────┐
           ├─ some → learn_recipe → measure → ocr_measured ┐
           └─ none → measure → ocr_measured ──────┤
                                                  ▼
                                         slot_and_table → export → END
```

Intake( unzip · collect )는 Sub-Graph로 묶어도 되고, 메인 그래프의 단순 노드로 둬도 된다.  
**조건부 엣지가 의미 있는 곳:** `filter_by_mag`, `detect_overlay` (및 measure 후 OCR 여부).

---



## 8. 기술 스택


| 레이어       | 기술                                                        |
| --------- | --------------------------------------------------------- |
| VLM / OCR | `gpt-4o` (+ 필요 시 전용 OCR)                                  |
| 구조화       | Pydantic                                                  |
| 워크플로우     | LangGraph StateGraph · conditional edges · (선택) Sub-Graph |
| 엣지 측정     | `sem_edge_tool.py` (`MeasureRecipe`, `measure_sem_batch`) |
| 테이블       | pandas                                                    |
| 환경        | Conda base · `.env` (`OPENAI_API_KEY`)                    |


---



## 9. 구현 범위



### Phase 1 (현재 노트북에 가깝게)

- 단일/다중 이미지 · VLM 추출 · slot · Long 테이블
- Edge tool 연동 · detect → overlay|edge 분기 (기존)
- 폴더 / URL / zip / 업로드 입력



### Phase 2 (이 기획서 본선)

- [x] 기획·구현 파일 분리 유지 
- [x] `parse_request`: `mag_mode` + `measure_goals` 입력 UX 
- [x] URL / zip intake 
- [x] `read_scale_bar` → `filter_by_mag` 순서 고정
- [x] overlay **all / some / none** 삼분 분기 
- [x] `ocr_existing` vs `ocr_measured` 노드 분리 
- [x] goals → recipe / edge / OCR 필터 연결 
- [x] Long 테이블에 `goal_id`, `source`, `magnification` 



### Phase 3+

- Streamlit 등 UI (파일 업로드 + 배율 라디오 + 목표 텍스트박스)
- unmatched · 매칭 리포트
- custom 배율 리스트 UI

---



## 10. 하지 않는 것 (당분간)

- 고정 Wide 스키마 (`top_width_nm` 전용 컬럼만)
- 숫자 나열 순서로 부위 묶기
- 전체 실루엣 패턴 분류를 1차 필수에 넣기
- 배율을 **항상** 최대만 쓰는 하드코딩 (반드시 옵션)

---



## 11. 한계 · 리스크


| 리스크                     | 대응                                            |
| ----------------------- | --------------------------------------------- |
| 스케일바 인식 실패              | px 유지 + warning; 수동 nm/px 입력 여지               |
| goals 자연어 애매            | 정규화 실패 시 전부 측정 후 느슨한 필터                       |
| some 케이스 학습 소스 선택       | 규칙 명시 + 로그                                    |
| 앵커 VLM 오차               | detail high · recipe 재사용 · unmatched 허용       |
| OCR이 직접 그린 선의 숫자를 잘못 읽음 | measure 시점 수치를 State에 같이 저장해 OCR과 cross-check |


---



## 12. 미결 (구현 직전 확정)

- [x] `measure_goals` 표준 id 목록 (`top_width`, `height`, …) 
- [ ] some일 때 학습 이미지 선정 규칙
- [ ] `ocr_measured` vs measure 수치 중 **최종 truth** (권장: measure 수치 primary, OCR은 검증)
- [ ] custom 배율 입력 UX (절대 배율 표기가 있는 장 한정 — 대부분 스케일 추정만 가능)
- [ ] 샘플 데이터 경로 (`data/metrology/` 로 일단 이동 완료, 운영 경로 확정 필요)

---



## 13. 산출물 · 문서 관계

프로젝트 루트: `실습폴더/AI agent practice/`


| 파일                                   | 역할                            |
| ------------------------------------ | ----------------------------- |
| **이 문서** (`TEM_SEM_측정_Agent_기획서.md`) | 시퀀스·분기·옵션·스키마 기획              |
| `AIagent_practice.ipynb`             | 실험·Phase 구현                   |
| `sem_edge_tool.py`                   | 엣지 측정 · recipe (**이미 Tool화**) |
| `data/metrology/`                    | 샘플 · uploads · output         |
| `../.env`                            | API 키 (상위 실습폴더)               |


---



## 14. Tool로 만들 항목

그래프 **노드(오케스트레이션)** 와 **Tool(재사용·단일 책임 함수)** 를 나눈다.  
노드는 분기·State 갱신만 하고, 무거운 입출력·CV·VLM 호출은 Tool로 둔다.

### 14.1 이미 Tool / 모듈인 것


| Tool · 모듈                                               | 상태           | 역할                                                          |
| ------------------------------------------------------- | ------------ | ----------------------------------------------------------- |
| `sem_edge_tool.measure_sem_structure`                   | ✅ 분리됨        | 단일 이미지 edge 측정                                              |
| `sem_edge_tool.measure_sem_batch`                       | ✅ 분리됨        | recipe 기반 배치 측정                                             |
| `MeasureRecipe` / `PatternRef`                          | ✅ 분리됨        | 학습된 측정 방식 재사용                                               |
| `tools.scale_bar.read_scale_bar`                        | ✅ 구현 (07-14) | classic/info_bar/derived_overlay 구분 · nm/px · mag           |
| `tools.scale_bar.filter_by_mag`                         | ✅ 구현 (07-14) | `max_only` / `all` / `custom`                               |
| `tools.overlay.classify_overlay_batch`                  | ✅ 구현 (07-14) | 측정값 오버레이 **all / some / none** 분기                           |
| `tools.overlay.route_overlay_mode`                      | ✅ 구현 (07-14) | LangGraph → ocr_existing / learn_and_measure / edge_measure |
| `tools.measure_source.plan_measure_sources`             | ✅ 구현 (07-14) | 삼분 결과 → **읽기 vs edge 생성** 이미지별 계획 · `source` 태그             |
| `tools.ocr_vlm.ocr_existing` / `ocr_existing_node`      | ✅ 구현 (07-14) | 원본 오버레이 VLM 읽기 · `source=ocr_existing`                      |
| `tools.ocr_vlm.ocr_measured` / `ocr_measured_node`      | ✅ 구현 (07-14) | 그린 측정 이미지 OCR(검증) · `source=ocr_measured`                   |
| `tools.recipe.parse_goals` / `filter_by_goals`          | ✅ 구현 (07-14) | 자연어·별칭 → 표준 goal · OCR/edge/table 필터                        |
| `tools.recipe.learn_recipe` / `edge_measure_with_goals` | ✅ 구현 (07-14) | goals → recipe.enabled_* · edge kinds 제한                    |




### 14.2 앞으로 Tool로 뺄 것 (권장)


| 후보 Tool 이름                  | 입력이 되는 시퀀스 | 왜 Tool인가                                 | 우선순위                                                   |
| --------------------------- | ---------- | ---------------------------------------- | ------------------------------------------------------ |
| `resolve_input_tool`        | ②          | URL 다운로드 / 로컬 경로 정규화                     | P1                                                     |
| `unzip_collect_tool`        | ③–⑤        | zip 해제 + 이미지 확장자 수집                      | P1                                                     |
| `read_scale_bar_tool`       | ⑥          | 스케일바·배율·nm/px 추출 (VLM 또는 CV)             | P1 → **구현됨** `tools/scale_bar.py`                      |
| `filter_by_mag_tool`        | ⑦          | `mag_mode`에 따라 path 리스트 필터               | P2                                                     |
| `detect_overlay_tool`       | ⑧          | 이미지별 오버레이 유무 → all/some/none             | P1 → **구현됨** `tools/overlay.py`                        |
| `plan_measure_sources_tool` | ⑧→⑨        | 삼분 결과 입력 → 읽기 vs edge 생성 · `source`      | P1 → **구현됨** `tools/measure_source.py`                 |
| `ocr_existing_tool`         | ⑨a         | 원본 측정 오버레이 → `Measurement[]`             | P1 → **구현됨** `tools/ocr_vlm.py`                        |
| `ocr_measured_tool`         | ⑪          | **그린** 측정 이미지 OCR (검증용)                  | P2 → **구현됨** `tools/ocr_vlm.py`                        |
| `parse_goals_tool`          | ①          | 자연어 `goal_text` → `measure_goals[]`      | P1 → **구현됨** `tools/recipe.py`                         |
| `parse_request` (모듈)        | ①          | mag_mode + goals → State 패치 · FastAPI UX | P2 → **구현됨** `tools/parse_request.py` + `parse_app.py` |
| `learn_recipe_tool`         | ⑨b         | 어노테이션 이미지(+goals) → `MeasureRecipe`      | P1 → **구현됨** `tools/recipe.py`                         |
| `filter_by_goals_tool`      | ⑨·⑫        | Measurement 리스트를 goals로 필터               | P2 → **구현됨** `tools/recipe.py`                         |
| `export_table_tool`         | ⑬          | Long / stats / pivot CSV 저장              | P2 → **구현됨** `tools/table.py` (long)                   |
| `assign_slot_tool`          | ⑫          | 앵커·orientation·goal → `slot_key`         | P1 → **구현됨** `tools/table.assign_slot_key`             |
| `align_slots_tool`          | ⑫          | 이미지 간 앵커 거리로 slot 정렬                     | P2                                                     |




### 14.3 Tool로 두지 않는 것 (노드에 남김)


| 항목                                         | 이유                  |
| ------------------------------------------ | ------------------- |
| `parse_request` (옵션 → State)               | 오케스트레이션·입력 바인딩      |
| `filter_by_mag` / `detect_overlay` **라우팅** | conditional edge 전용 |
| 전체 Graph compile · invoke                  | Agent 엔트리           |




### 14.4 권장 패키지 구조 (차후)

```
AI agent practice/
├── AIagent_practice.ipynb      # 그래프 조립 · 데모
├── TEM_SEM_측정_Agent_기획서.md
├── sem_edge_tool.py            # ✅ edge
├── parse_app.py                # ✅ FastAPI 껍데기 (17일차 Notes 패턴)
├── parse_request.html          # ✅ 브라우저 UX · /docs
└── tools/
    ├── __init__.py
    ├── scale_bar.py            # ✅ read_scale_bar, filter_by_mag
    ├── overlay.py              # ✅ classify_overlay_batch (all/some/none)
    ├── measure_source.py       # ✅ 읽기(ocr) vs edge 생성 분기 (삼분 결과 입력)
    ├── intake.py               # (예정) resolve / unzip / collect
    ├── ocr_vlm.py              # ✅ ocr_existing, ocr_measured
    ├── recipe.py               # ✅ parse_goals, filter_by_goals, learn_recipe, edge_measure_*
    ├── parse_request.py        # ✅ mag_mode + goals → RequestOptions · State 패치
    └── table.py                # ✅ Long 컬럼 · measurements_to_rows · enrich · export
```

LangChain/LangGraph `@tool` 래핑은 **노트북에서 bind**하고, 실제 로직은 `tools/*.py`에 둔다.  
브라우저 확인: `uvicorn parse_app:app --reload --port 8001` → [http://127.0.0.1:8001/](http://127.0.0.1:8001/)

---



## 15. 날짜별 수정사항 요약



### 2026-07-10 — 기획 v1 초안

- Agent 한 줄 정의: `image_path` → VLM 측정값 추출 → 앵커 기준 묶음 → Long 테이블
- “같은 묶음” = 앵커 위치 + 측정 방향 (숫자 나열 순서 금지)
- 스키마: `Measurement`, `ImageExtraction`, 초안 `MetrologyState`
- LangGraph 선형 흐름: `extract → slot → append → export`
- Phase 1 범위·하지 않는 것·리스크(앵커 오차 등) 정리
- *(당시 위치:* `AIagent_practice.ipynb` *상단 마크다운)*



### 2026-07-13 — Phase 1 구현 · edge 분리

- 노트북에 Graph·VLM overlay·slot·집계·CSV 저장 구현
- 입력 확장: 폴더 / 파일 / URL / **zip** / 업로드 (`collect_image_paths`, `run_from_source`)
- **edge 측정** 추가 후 `sem_edge_tool.py`로 분리 (`MeasureRecipe`, 배치 측정)
- `detect → overlay | edge` 이분 조건부 분기
- 산출물: `data/metrology/output/*.csv`, edge overlay PNG



### 2026-07-14 — 기획 v2 · 옵션 path · 문서/폴더 정리

- 기획서를 `TEM_SEM_측정_Agent_기획서.md`로 **분리**, 노트북은 구현 전용
- 전체 시퀀스 재작성: intake → **scale_bar → mag 필터** → overlay 삼분 → OCR/측정
- `mag_mode` (`max_only` / `all` / `custom`) 시작 옵션 path 추가
- `measure_goals` **/** `goal_text` — 측정 부위 목표를 OCR·recipe·edge에 반영
- OCR 노드 분리: `ocr_existing` vs `ocr_measured`; truth는 measure primary 권장
- 오버레이 **all / some / none** 삼분 분기
- 프로젝트 폴더 `AI agent practice/`로 관련 파일·`data/metrology` 이동
- 본 문서: 섹션별 날짜 주석 + Tool 후보 목록(§14) + 날짜별 요약(§15)
- `tools/ocr_vlm.py`**:** `ocr_existing` / `ocr_measured` Tool + LangGraph 노드 헬퍼 구현
- `tools/recipe.py`**:** goals 표준 id · parse · filter · learn_recipe · edge kinds 연결
- `tools/table.py`**:** Long 스키마에 `goal_id` · `source` · `magnification` 고정
- `tools/parse_request.py` **+** `parse_app.py`**:** mag_mode · measure_goals 입력 UX · FastAPI HTML 껍데기 (17일차)
- **배율:** 이미지에 x 표기 없음 전제 → `estimate_mag_from_scale(nm/px)` 상대 추정 · `max_only` 비교

---

*기획서 v2.6 — 2026-07-14 · 배율=스케일바 추정 · parse_request UX*