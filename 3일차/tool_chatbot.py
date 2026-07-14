import json
import os
from pathlib import Path

import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to weather and stock tools. "
    "Use tools when needed. Answer in Korean."
)


def get_env_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / ".env",
        script_dir / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            return path
    return script_dir.parent / ".env"


@st.cache_resource
def get_client() -> OpenAI:
    env_path = get_env_path()
    if not env_path.exists() or env_path.stat().st_size == 0:
        st.error(f".env 파일을 확인하세요: {env_path}")
        st.stop()
    load_dotenv(env_path, override=True, encoding="utf-8-sig")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY가 .env에 없습니다.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_weather(city: str) -> str:
    """도시 날씨 조회 (wttr.in API)"""
    try:
        response = requests.get(
            f"https://wttr.in/{city.strip()}",
            params={"format": "j1"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)

        return json.dumps(
            {
                "city": area_name,
                "temperature_c": current.get("temp_C"),
                "feels_like_c": current.get("FeelsLikeC"),
                "humidity": current.get("humidity"),
                "weather": current.get("weatherDesc", [{}])[0].get("value"),
                "wind_kmph": current.get("windspeedKmph"),
                "source": "wttr.in",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_us_stock_price(ticker: str) -> str:
    """미국 주식 시가·종가 조회 (yfinance)"""
    symbol = ticker.strip().upper()
    try:
        hist = yf.Ticker(symbol).history(period="2d")
        if hist.empty:
            return json.dumps({"error": f"{symbol} 데이터가 없습니다."}, ensure_ascii=False)

        latest = hist.iloc[-1]
        return json.dumps(
            {
                "ticker": symbol,
                "open": round(float(latest["Open"]), 2),
                "close": round(float(latest["Close"]), 2),
                "currency": "USD",
                "source": "yfinance",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "도시의 현재 날씨(기온, 체감, 습도, 날씨 상태)를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "도시명 (예: Seoul, Tokyo)"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_price",
            "description": "미국 주식 티커(AAPL, TSLA 등)의 최근 시가·종가를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "주식 티커 심볼"}},
                "required": ["ticker"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_us_stock_price": get_us_stock_price,
}


def run_agent(user_message: str, history: list[dict], temperature: float) -> tuple[str, list[dict]]:
    """Tool Calling 루프 — tool 호출이 없을 때까지 반복"""
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": answer})
            return answer, history

        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments or "{}")
            result = TOOL_FUNCTIONS[fn_name](**fn_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


def main() -> None:
    st.set_page_config(page_title="날씨·주식 챗봇", page_icon="🤖")
    st.title("🤖 날씨 · 주식 Tool Calling 챗봇")
    st.caption("3.4 Tool_Calling 실습2 · Streamlit")

    with st.sidebar:
        st.header("설정")
        temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.1)
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_history = []
            st.rerun()
        st.markdown("**사용 가능 Tool**")
        st.markdown("- `get_weather` — 도시 날씨")
        st.markdown("- `get_us_stock_price` — 미국 주식")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("날씨나 주식에 대해 물어보세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Tool Calling 중..."):
                try:
                    answer, st.session_state.agent_history = run_agent(
                        prompt,
                        st.session_state.agent_history,
                        temperature,
                    )
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
