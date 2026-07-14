"""
FastAPI + Slack Event Subscription + LangChain Agent (SKHY LSTM)

일반 대화: gpt-5-nano
모델 설명/test 평가: LangChain tools
응답: Slack chat.postMessage
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from dl_tools import MODEL_TOOLS

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 대화 기록: 최근 N턴(사람+AI)만 유지
MAX_HISTORY_MESSAGES = 20

client = WebClient(token=SLACK_BOT_TOKEN)
app = FastAPI(title="SKHY LSTM Slack Agent")
_seen: set[str] = set()
_executor = ThreadPoolExecutor(max_workers=4)
_histories: dict[str, list[BaseMessage]] = {}
_history_lock = threading.Lock()

SYSTEM_PROMPT = """당신은 Slack에서 동작하는 SKHY 회로 시계열(LSTM) 예측 도우미입니다.

역할:
1. 일반 대화는 친절하고 간결한 한국어로 답하세요.
2. 이전 대화 내용(이름, 선호, 앞서 한 질문)을 기억하고 이어서 답하세요.
3. 모델이 무엇인지/어떻게 쓰는지 물으면 explain_model_usage 도구를 사용하세요.
4. 예측, test 평가, 정확도, 성능, 얼마나 잘 맞는지 물으면 evaluate_test_answer 도구를 사용하세요.
5. 사용자가 직접 TIME/Input_V 숫자를 넣을 필요는 없습니다. test_answer 구간을 모델이 예측·평가합니다.
6. 도구 결과는 숫자 나열이 아니라, 잘된 점/아쉬운 점/채널별 차이를 쉽게 해석해 전달하세요.
7. Slack이므로 핵심만 짧게 정리하세요.
"""

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.3, api_key=OPENAI_API_KEY)
agent = create_react_agent(llm, MODEL_TOOLS, prompt=SYSTEM_PROMPT)


def verify_slack_signature(body: bytes, timestamp: str | None, signature: str | None) -> None:
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="missing signature headers")
    if abs(time.time() - int(timestamp)) > 60 * 5:
        raise HTTPException(status_code=401, detail="stale request")
    base = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        base.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(f"v0={digest}", signature):
        raise HTTPException(status_code=401, detail="invalid signature")


def _session_key(channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'root'}"


def _message_text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def run_agent(session_key: str, user_text: str) -> str:
    with _history_lock:
        history = list(_histories.get(session_key, []))

    result = agent.invoke(
        {"messages": history + [HumanMessage(content=user_text)]}
    )
    messages = result.get("messages") or []

    answer = "응답을 생성하지 못했습니다. 다시 시도해 주세요."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            text = _message_text(msg)
            if text:
                answer = text
                break

    # 툴 중간 메시지는 버리고, 사람/최종 AI만 세션에 저장
    with _history_lock:
        updated = history + [
            HumanMessage(content=user_text),
            AIMessage(content=answer),
        ]
        _histories[session_key] = updated[-MAX_HISTORY_MESSAGES:]

    return answer


def reply_in_background(channel: str, text: str, thread_ts: str | None = None) -> None:
    session_key = _session_key(channel, thread_ts)
    try:
        answer = run_agent(session_key, text)
        print("Agent 응답:", answer[:200])
        kwargs = {"channel": channel, "text": answer}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        client.chat_postMessage(**kwargs)
    except SlackApiError as e:
        print("postMessage 실패:", e.response.get("error"))
    except Exception as e:
        print("Agent 처리 실패:", e)
        try:
            client.chat_postMessage(
                channel=channel,
                text=f"처리 중 오류가 발생했습니다: {e}",
                **({"thread_ts": thread_ts} if thread_ts else {}),
            )
        except Exception:
            pass


@app.get("/health")
def health():
    model_ok = (Path(__file__).resolve().parent / "models" / "skhy_lstm.pt").exists()
    return {"ok": True, "model_loaded": model_ok}


@app.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
):
    body = await request.body()
    verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature)
    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    if payload.get("type") != "event_callback":
        return {"ok": True}

    event_id = payload.get("event_id")
    if event_id:
        if event_id in _seen:
            return {"ok": True}
        _seen.add(event_id)
        if len(_seen) > 5000:
            _seen.clear()

    event = payload.get("event") or {}
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return {"ok": True}

    event_type = event.get("type")
    if event_type == "message" and event.get("channel_type") != "im":
        return {"ok": True}
    if event_type not in ("app_mention", "message"):
        return {"ok": True}

    text = (event.get("text") or "").strip()
    channel = event.get("channel")
    if not text or not channel:
        return {"ok": True}

    if event_type == "app_mention":
        parts = text.split(maxsplit=1)
        text = parts[1] if len(parts) > 1 else text

    thread_ts = None if event.get("channel_type") == "im" else (event.get("thread_ts") or event.get("ts"))
    print("받은 메시지:", text)
    _executor.submit(reply_in_background, channel, text, thread_ts)

    return {"ok": True}
