"""
parse_request FastAPI 껍데기 (17일차 Notes 패턴).

브라우저 UX로 mag_mode · measure_goals 입력을 받아
``tools.parse_request`` 로 State를 정규화한 뒤 JSON으로 보여준다.

실행 (day17 또는 base)::

    cd "AI agent practice"
    uvicorn parse_app:app --reload --port 8001

- UI:   http://127.0.0.1:8001/
- docs: http://127.0.0.1:8001/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tools.parse_request import (
    ParseRequestInput,
    ParseRequestResult,
    list_goal_catalog,
    parse_request,
    summarize_parse,
)

app = FastAPI(
    title='TEM/SEM parse_request',
    description='mag_mode + measure_goals 입력 UX · State 초기화 미리보기',
    version='0.1.0',
)
HTML_PATH = Path(__file__).with_name('parse_request.html')


class ParseBody(BaseModel):
    """POST /parse JSON body."""

    input_uri: str = Field(default='', description='파일/폴더/URL/zip')
    mag_mode: str = Field(default='all', description='max_only | all | custom')
    mag_custom: list[float] = Field(default_factory=list)
    mag_custom_text: str = Field(
        default='',
        description='custom 배율 문자열 예: 50k, 100000',
    )
    measure_goals: list[str] = Field(default_factory=list)
    goal_text: str = Field(
        default='',
        description='자연어 목표 예: 기둥 상단 폭과 높이만. 각도 제외',
    )


@app.get('/')
def home():
    """브라우저에서 parse_request UX."""
    return FileResponse(HTML_PATH)


@app.get('/api/goals')
def api_goals():
    """표준 measure_goals 카탈로그 (체크박스용)."""
    return {'goals': list_goal_catalog(), 'count': len(list_goal_catalog())}


@app.get('/api/mag-modes')
def api_mag_modes():
    return {
        'modes': [
            {
                'value': 'all',
                'label': '전체 (all)',
                'hint': '배율 필터 없이 수집된 이미지 전부 측정',
            },
            {
                'value': 'max_only',
                'label': '최대 배율만 (max_only)',
                'hint': '이미지에 배율 숫자는 없음 → 스케일바 nm/px로 상대 배율 추정 후 최대만',
            },
            {
                'value': 'custom',
                'label': '지정 배율 (custom, 고급)',
                'hint': '정보바에 절대 배율(50kx 등)이 적힌 경우에만. 대부분 스케일 추정만 가능하면 비움',
            },
        ],
        'note': (
            '배율(x)은 이미지에 안 나오는 경우가 많다. '
            'read_scale_bar가 스케일바로부터 nm/px → 상대 배율을 추정하고, '
            'mag_mode는 그 추정값으로 필터한다.'
        ),
    }


@app.post('/parse', response_model=ParseRequestResult)
def api_parse(body: ParseBody) -> ParseRequestResult:
    """입력 UX → parse_request → State 패치."""
    result = parse_request(
        input_uri=body.input_uri,
        mag_mode=body.mag_mode,
        mag_custom=body.mag_custom,
        mag_custom_text=body.mag_custom_text,
        measure_goals=body.measure_goals,
        goal_text=body.goal_text,
    )
    return result


@app.post('/parse/form', response_model=ParseRequestResult)
def api_parse_form(payload: ParseRequestInput) -> ParseRequestResult:
    """OpenAPI /docs 에서 쓰기 쉬운 동일 엔드포인트."""
    return parse_request(
        input_uri=payload.input_uri,
        mag_mode=payload.mag_mode,
        mag_custom=payload.mag_custom,
        mag_custom_text=payload.mag_custom_text,
        measure_goals=payload.measure_goals,
        goal_text=payload.goal_text,
    )


@app.get('/health')
def health():
    return {'ok': True, 'service': 'parse_request'}


if __name__ == '__main__':
    demo = parse_request(
        input_uri='data/metrology',
        mag_mode='max_only',
        goal_text='기둥 상단 폭과 높이만. 각도 제외',
    )
    print(summarize_parse(demo))
    print(demo.model_dump_json(indent=2))
