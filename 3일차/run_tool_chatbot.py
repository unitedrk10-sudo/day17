"""Streamlit 실행 래퍼 - Windows SSL 인증서 저장소 오류 우회."""
import os
import ssl
import sys

import certifi


def _create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile or certifi.where())
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


ssl.create_default_context = _create_default_context

if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    sys.argv = [
        "streamlit",
        "run",
        "tool_chatbot.py",
        "--server.headless",
        "true",
        *sys.argv[1:],
    ]
    from streamlit.web import cli as stcli

    stcli.main()
