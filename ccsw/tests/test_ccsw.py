# -*- coding: utf-8 -*-
"""ccsw 테스트 — 임시 폴더만 사용 (실제 ~/.claude·환경변수·레지스트리는 건드리지 않음)."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ccsw  # noqa: E402

K1 = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAwork1"
K2 = "sk-ant-api03-BBBBBBBBBBBBBBBBBBBBpers2"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CCSW_HOME", str(tmp_path / "ccsw"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    for v in ccsw.KEY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(ccsw, "_windows_env", lambda: [])
    monkeypatch.setattr(ccsw, "_vscode_settings", lambda: [])
    settings = tmp_path / "claude" / "settings.json"
    settings.parent.mkdir()
    return tmp_path, settings


def run(*args, stdin=None, env_extra=None):
    """실제 helper처럼 별도 프로세스로 실행."""
    e = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, os.path.join(HERE, "ccsw.py"), *args], input=stdin,
                          capture_output=True, text=True, env=e)


def test_migrate_install_switch_and_helper(env, capsys):
    tmp, settings = env
    settings.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": K1, "ANTHROPIC_BASE_URL": "https://gw.example"},
                                    "model": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit):                                   # env 키가 있으면 --migrate 없이 거부, 파일 그대로
        ccsw.main(["install"])
    assert "ANTHROPIC_API_KEY" in settings.read_text(encoding="utf-8")
    ccsw.main(["install", "--migrate", "work", "--ttl", "30000"])
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert "ANTHROPIC_API_KEY" not in data["env"] and data["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example"
    assert data["env"][ccsw.TTL_VAR] == "30000" and data["model"] == "x" and "ccsw.py" in data["apiKeyHelper"]
    assert list(tmp.joinpath("claude").glob("settings.json.bak-*"))                     # 백업
    assert os.path.isfile(os.path.join(ccsw.home(), "ccsw.py"))                         # 스크립트 복사
    # helper 명령을 그대로 실행하면 활성 키가 나온다 (환경변수 CCSW_HOME 없이도 — --home 고정)
    e = {k: v for k, v in os.environ.items() if k != "CCSW_HOME"}
    r = subprocess.run(data["apiKeyHelper"], shell=True, capture_output=True, text=True, env=e)
    assert r.returncode == 0 and r.stdout.strip() == K1
    r = run("add", "personal", "--stdin", stdin=K2 + "\n")
    assert r.returncode == 0 and K2 not in r.stdout                                    # 키는 가려서 표시
    ccsw.main(["use", "personal"])
    r = subprocess.run(data["apiKeyHelper"], shell=True, capture_output=True, text=True, env=e)
    assert r.stdout.strip() == K2                                                      # 전환 즉시 helper 출력 바뀜
    out = capsys.readouterr().out
    assert K1 not in out and K2 not in out


def test_doctor_flags_overrides(env, monkeypatch, capsys):
    tmp, settings = env
    run("add", "a", "--stdin", stdin=K1 + "\n")
    ccsw.main(["install"])
    assert ccsw.main(["doctor"]) == 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")                                      # 셸에 남은 키 → FAIL
    assert ccsw.main(["doctor"]) == 1
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    proj = tmp / "proj" / ".claude"
    proj.mkdir(parents=True)
    (proj / "settings.local.json").write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "y"}}), encoding="utf-8")
    assert ccsw.main(["doctor", "--project", str(tmp / "proj")]) == 1
    out = capsys.readouterr().out
    assert K1 not in out and "settings.local.json" in out


def test_get_fails_cleanly_and_guards(env):
    tmp, settings = env
    r = run("get")
    assert r.returncode == 1 and r.stdout == ""                                      # 활성 키 없음 → 출력 없이 실패
    run("add", "a", "--stdin", stdin=K1 + "\n")
    assert run("add", "a", "--stdin", stdin=K2 + "\n").returncode != 0                # 덮어쓰기는 --force 필요
    assert run("remove", "a").returncode != 0                                         # 활성 키 삭제 금지
    assert run("add", "bad name", "--stdin", stdin=K1).returncode != 0
    assert run("add", "b", "--stdin", stdin="has space\n").returncode != 0
    settings.write_text("{ broken", encoding="utf-8")
    assert run("install").returncode != 0 and settings.read_text(encoding="utf-8") == "{ broken"   # 깨진 JSON은 안 건드림


def test_uninstall_removes_helper_only(env):
    tmp, settings = env
    run("add", "a", "--stdin", stdin=K1 + "\n")
    ccsw.main(["install"])
    ccsw.main(["uninstall"])
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert "apiKeyHelper" not in data and ccsw.TTL_VAR not in data.get("env", {})
    assert ccsw.read_key("a") == K1


def test_clipboard_and_bad_paste(env, monkeypatch, capsys):
    monkeypatch.setattr(ccsw, "read_clipboard", lambda: K1 + "\r\n")
    assert ccsw.main(["add", "clip", "--clipboard"]) == 0 and ccsw.read_key("clip") == K1
    assert K1 not in capsys.readouterr().out
    for bad in ("\x16", "sk-ant-\x16abcdefghijklmnopqrstu", "short", "sk-ant-한글abcdefghijklmnopqrstu"):
        monkeypatch.setattr(ccsw, "read_clipboard", lambda bad=bad: bad)
        with pytest.raises(SystemExit):
            ccsw.main(["add", "x", "--clipboard"])
    assert ccsw.read_key("x") is None


def test_prompt_ctrl_v_falls_back_to_clipboard(env, monkeypatch, tmp_path):
    monkeypatch.setattr(ccsw.getpass, "getpass", lambda prompt="": "\x16")        # 옛 콘솔: Ctrl+V가 ^V로
    monkeypatch.setattr(ccsw, "read_clipboard", lambda: K2)
    assert ccsw.main(["add", "p"]) == 0 and ccsw.read_key("p") == K2
    f = tmp_path / "k.txt"
    f.write_text("\ufeff" + K1 + "\r\n", encoding="utf-8")                      # 메모장 BOM·CRLF
    assert ccsw.main(["add", "f", "--file", str(f)]) == 0 and ccsw.read_key("f") == K1
