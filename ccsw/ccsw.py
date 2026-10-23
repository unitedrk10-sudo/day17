#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ccsw — Claude Code API 키를 이름으로 전환 (실행 중인 세션·VS Code 사이드바 포함).

원리: Claude Code의 apiKeyHelper 설정이 이 스크립트(`ccsw.py get`)를 실행해 키를 받는다.
CLAUDE_CODE_API_KEY_HELPER_TTL_MS 주기마다 다시 받으므로, `ccsw use <이름>`으로 활성 키만 바꾸면
실행 중인 모든 세션이 재시작 없이 새 키를 쓴다. 표준 라이브러리만 사용 (Windows 우선, macOS·Linux 동작).

명령
  install [--migrate 이름] [--ttl 60000]   settings.json에 apiKeyHelper 설정 (백업 후), 스크립트를 ~/.claude/ccsw/로 복사
                                            --migrate: settings.json env의 기존 키를 그 이름으로 옮기고 env에서 삭제
  add 이름 [--clipboard | --file 경로 | --stdin | --from-env 변수]  키 등록 (기본: 화면에 안 보이게 입력, --clipboard: 복사해 둔 키 바로 읽기)
  use 이름                                  활성 키 전환 (+ settings.json 다시 쓰기로 실행 중 세션에 즉시 반영 유도)
  list | current                            등록 키 목록 / 현재 활성 키 (키는 가려서 표시)
  remove 이름                               키 삭제 (활성 키는 삭제 불가)
  doctor                                    helper를 무시하게 만드는 남은 환경변수·설정 점검
  get                                       helper 전용: 활성 키 원문 출력 (사람이 쓸 일 없음)
  uninstall                                 settings.json에서 apiKeyHelper·TTL 제거 (백업 후, 키 파일은 남김)

경로: 저장소 = $CCSW_HOME 또는 ~/.claude/ccsw (keys/<이름>.key, active)
      설정   = --settings 또는 $CLAUDE_CONFIG_DIR/settings.json 또는 ~/.claude/settings.json
"""
import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
TTL_VAR = "CLAUDE_CODE_API_KEY_HELPER_TTL_MS"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


# ---------------------------------------------------------------- 경로
def claude_dir():
    return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")


def home():
    return os.environ.get("CCSW_HOME") or os.path.join(os.path.expanduser("~"), ".claude", "ccsw")


def keys_dir():
    return os.path.join(home(), "keys")


def key_path(name):
    return os.path.join(keys_dir(), f"{name}.key")


def active_path():
    return os.path.join(home(), "active")


def settings_path(arg=None):
    return arg or os.path.join(claude_dir(), "settings.json")


# ---------------------------------------------------------------- 유틸
def mask(key):
    key = key or ""
    return "(비어 있음)" if not key else (key[:7] + "…" + key[-4:] if len(key) > 14 else "…" + key[-2:])


def check_name(name):
    if not NAME_RE.match(name or ""):
        raise SystemExit(f"이름은 영문·숫자·_ . - 1~40자: {name!r}")
    return name


def atomic_write(path, text, private=False):
    """같은 폴더 임시 파일 → os.replace (읽는 쪽이 반쪽 파일을 보지 않게)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ccsw-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        if private and os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if private:
        restrict(path)


def restrict(path):
    """키 파일을 본인 계정만 읽게 (Windows: icacls 상속 제거 + 본인만, 실패하면 경고만)."""
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    user = os.environ.get("USERNAME")
    if not user:
        return
    try:
        subprocess.run(["icacls", path, "/inheritance:r", "/grant:r", f"{user}:F"],
                       capture_output=True, check=True, timeout=15)
    except Exception:
        print(f"경고: 권한 제한(icacls) 실패 — {path} 접근 권한을 직접 확인하세요", file=sys.stderr)


def read_key(name):
    p = key_path(name)
    if not os.path.isfile(p):
        return None
    return open(p, encoding="utf-8").read().strip() or None


def active_name():
    p = active_path()
    return open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else None


def names():
    d = keys_dir()
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".key")) if os.path.isdir(d) else []


def load_settings(path):
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8").read()
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"settings.json이 올바른 JSON이 아닙니다 ({e}) — 고친 뒤 다시 실행 (아무것도 바꾸지 않았음)")


def save_settings(path, data, backup=True):
    if backup and os.path.isfile(path):
        b = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, b)
        print(f"백업: {b}")
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def helper_command(script, python=None):
    """helper 명령 — 저장소 경로(--home)를 명령에 고정해 Claude Code 쪽 환경변수와 무관하게."""
    q = lambda p: '"' + p.replace("\\", "/") + '"'
    return f"{q(python or sys.executable)} -I {q(script)} --home {q(home())} get"


# ---------------------------------------------------------------- 명령
def cmd_get(a):
    n = active_name()
    key = read_key(n) if n else None
    if not key:
        sys.stderr.write("ccsw: 활성 키 없음 — `ccsw use <이름>`\n")
        return 1
    sys.stdout.write(key + "\n")
    return 0


def read_clipboard():
    """클립보드 텍스트 (Windows: PowerShell Get-Clipboard, macOS: pbpaste, Linux: wl-paste/xclip)."""
    if os.name == "nt":
        cmds = [["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"]]
    elif sys.platform == "darwin":
        cmds = [["pbpaste"]]
    else:
        cmds = [["wl-paste", "--no-newline"], ["xclip", "-o", "-selection", "clipboard"]]
    for c in cmds:
        try:
            r = subprocess.run(c, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return r.stdout
        except (OSError, subprocess.SubprocessError):
            continue
    raise SystemExit("클립보드를 읽지 못했습니다 — --stdin 또는 입력 프롬프트를 쓰세요")


def check_key(key):
    """키는 공백·제어문자 없는 ASCII. 콘솔이 Ctrl+V를 붙여넣기 대신 글자(^V)로 넣은 경우를 여기서 막는다."""
    if not key:
        raise SystemExit("키가 비었습니다")
    bad = [c for c in key if not 33 <= ord(c) < 127]
    if bad:
        hint = " — Ctrl+V가 붙여넣기가 아니라 글자로 들어갔습니다. 오른쪽 클릭 붙여넣기나 --clipboard를 쓰세요" \
            if any(ord(c) < 32 for c in bad) else ""
        raise SystemExit(f"키에 공백·제어문자·비ASCII 문자가 {len(bad)}개 있습니다{hint}")
    if len(key) < 20:
        raise SystemExit(f"키가 너무 짧습니다({len(key)}자) — 일부만 붙여넣어졌는지 확인")
    return key


def cmd_add(a):
    name = check_name(a.name)
    if a.stdin:
        key = sys.stdin.readline().strip()
    elif a.from_env:
        key = (os.environ.get(a.from_env) or "").strip()
    elif a.clipboard:
        key = read_clipboard().strip()
    elif a.file:
        raw = open(a.file, "rb").read()
        key = (raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")).strip()  # 메모장 UTF-16·BOM
    else:
        print("키를 복사해 둔 상태면 Ctrl+V 한 번 + Enter (글자는 안 보이는 게 정상)")
        key = getpass.getpass(f"'{name}' 키 입력 (화면에 표시 안 됨): ").strip()
        if any(ord(c) < 32 for c in key):
            # 옛 콘솔은 Ctrl+V·Ctrl+Shift+V를 붙여넣기 대신 제어문자(^V)로 넘긴다 → 클립보드에서 직접 읽는다
            try:
                clip = read_clipboard().strip()
            except SystemExit:
                clip = ""
            if clip and all(33 <= ord(c) < 127 for c in clip) and len(clip) >= 20:
                print(f"붙여넣기가 글자(^V)로 들어와 클립보드에서 직접 읽었습니다: {mask(clip)}")
                key = clip
    check_key(key)
    if os.path.isfile(key_path(name)) and not a.force:
        raise SystemExit(f"'{name}' 이미 있음 — 덮어쓰려면 --force")
    atomic_write(key_path(name), key + "\n", private=True)
    print(f"등록: {name} = {mask(key)}")
    if not active_name():
        atomic_write(active_path(), name + "\n")
        print(f"활성 키가 없어 '{name}'을 활성으로 설정")
    return 0


def nudge(settings):
    """settings.json을 같은 내용으로 다시 써서 실행 중인 세션이 설정(apiKeyHelper 포함)을 다시 읽게 유도."""
    if os.path.isfile(settings):
        data = load_settings(settings)
        if data.get("apiKeyHelper"):
            save_settings(settings, data, backup=False)


def cmd_use(a):
    name = check_name(a.name)
    key = read_key(name)
    if not key:
        raise SystemExit(f"'{name}' 키 없음 — 등록된 이름: {', '.join(names()) or '(없음)'}")
    atomic_write(active_path(), name + "\n")
    if not a.no_nudge:
        nudge(settings_path(a.settings))
    ttl = load_settings(settings_path(a.settings)).get("env", {}).get(TTL_VAR, "?")
    print(f"활성 키 → {name} ({mask(key)}). 실행 중인 세션은 늦어도 TTL({ttl}ms) 안에 새 키 사용")
    return 0


def cmd_list(a):
    act = active_name()
    if not names():
        print("(등록된 키 없음) — `ccsw add <이름>`")
    for n in names():
        print(f"{'*' if n == act else ' '} {n:20s} {mask(read_key(n))}")
    return 0


def cmd_current(a):
    n = active_name()
    print(f"{n} ({mask(read_key(n))})" if n else "(활성 키 없음)")
    return 0 if n and read_key(n) else 1


def cmd_remove(a):
    name = check_name(a.name)
    if name == active_name():
        raise SystemExit("활성 키는 삭제할 수 없습니다 — 다른 키로 `use` 후 삭제")
    if not os.path.isfile(key_path(name)):
        raise SystemExit(f"'{name}' 없음")
    os.remove(key_path(name))
    print(f"삭제: {name}")
    return 0


def cmd_install(a):
    settings = settings_path(a.settings)
    data = load_settings(settings)
    env = data.get("env") or {}
    present = [v for v in KEY_VARS if env.get(v)]
    if present and not a.migrate:
        raise SystemExit(f"settings.json env에 {', '.join(present)}가 있습니다 — helper보다 우선이라 그대로 두면 전환이 안 됩니다.\n"
                         f"`ccsw install --migrate <이름>`으로 그 키를 <이름>으로 옮기고 env에서 지우세요 (아무것도 바꾸지 않았음)")
    if a.migrate:
        name = check_name(a.migrate)
        if not present:
            raise SystemExit("--migrate: settings.json env에 옮길 키가 없습니다")
        if len({env[v] for v in present}) > 1:
            raise SystemExit(f"{' / '.join(present)} 값이 서로 다릅니다 — 하나만 남기고 다시 실행")
        if os.path.isfile(key_path(name)) and not a.force:
            raise SystemExit(f"'{name}' 이미 있음 — 덮어쓰려면 --force")
        atomic_write(key_path(name), env[present[0]].strip() + "\n", private=True)
        atomic_write(active_path(), name + "\n")
        for v in present:
            env.pop(v)
        print(f"이전: settings.json env의 {', '.join(present)} → 키 '{name}' ({mask(read_key(name))}), 활성으로 설정")
    # 스크립트를 저장소 폴더로 복사 — 리포 위치가 바뀌어도 helper가 깨지지 않게
    dst = os.path.join(home(), "ccsw.py")
    if os.path.abspath(__file__) != os.path.abspath(dst):
        os.makedirs(home(), exist_ok=True)
        shutil.copy2(os.path.abspath(__file__), dst)
    if os.name == "nt":                       # 짧게 부르기: %USERPROFILE%\.claude\ccsw\ccsw.cmd use work (PATH에 넣으면 ccsw use work)
        py = (a.python or sys.executable).replace("/", "\\")
        atomic_write(os.path.join(home(), "ccsw.cmd"),
                     f'@echo off\r\n"{py}" -I "{dst}" --home "{home()}" %*\r\n')
    data["apiKeyHelper"] = helper_command(dst, a.python)
    env[TTL_VAR] = str(a.ttl)
    data["env"] = env
    save_settings(settings, data)
    print(f"설정: {settings}\n  apiKeyHelper = {data['apiKeyHelper']}\n  env.{TTL_VAR} = {a.ttl}")
    if not active_name():
        print("다음: `ccsw add <이름>` → `ccsw use <이름>`")
    print("중요: 터미널·VS Code의 Claude Code 세션을 이번 한 번만 모두 닫았다 다시 여세요 "
          "(이미 떠 있는 세션은 예전 env 키를 기억합니다). 이후 전환은 재시작 없이 됩니다. 그 다음 `ccsw doctor`.")
    return 0


def cmd_uninstall(a):
    settings = settings_path(a.settings)
    data = load_settings(settings)
    data.pop("apiKeyHelper", None)
    if isinstance(data.get("env"), dict):
        data["env"].pop(TTL_VAR, None)
        if not data["env"]:
            data.pop("env")
    save_settings(settings, data)
    print("apiKeyHelper·TTL 제거 (키 파일은 그대로). 세션을 다시 열면 이전 방식(로그인 또는 env 키)으로 동작")
    return 0


def _windows_env():
    """Windows 사용자·시스템 환경변수 중 KEY_VARS (레지스트리 — 새로 여는 프로그램이 물려받는 값)."""
    out = []
    if os.name != "nt":
        return out
    import winreg
    spots = [(winreg.HKEY_CURRENT_USER, r"Environment", "사용자 환경변수"),
             (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "시스템 환경변수")]
    for root, sub, label in spots:
        try:
            with winreg.OpenKey(root, sub) as k:
                for v in KEY_VARS:
                    try:
                        winreg.QueryValueEx(k, v)
                        out.append(f"{label} {v}")
                    except FileNotFoundError:
                        pass
        except OSError:
            pass
    return out


def _vscode_settings():
    cands = []
    if os.name == "nt" and os.environ.get("APPDATA"):
        for app in ("Code", "Code - Insiders", "Cursor"):
            cands.append(os.path.join(os.environ["APPDATA"], app, "User", "settings.json"))
    elif sys.platform == "darwin":
        cands.append(os.path.expanduser("~/Library/Application Support/Code/User/settings.json"))
    else:
        cands.append(os.path.expanduser("~/.config/Code/User/settings.json"))
    return [p for p in cands if os.path.isfile(p)]


def doctor(settings, project=None):
    """반환 [(상태 OK|WARN|FAIL, 메시지)] — 키 값은 출력하지 않는다."""
    res = []
    data = load_settings(settings) if os.path.isfile(settings) else {}
    helper = data.get("apiKeyHelper")
    res.append(("OK", f"apiKeyHelper 설정됨 ({settings})") if helper and "ccsw" in helper
               else ("FAIL", f"apiKeyHelper 없음 또는 ccsw 아님 ({settings}) — `ccsw install`"))
    ttl = (data.get("env") or {}).get(TTL_VAR)
    res.append(("OK", f"{TTL_VAR} = {ttl}") if ttl else ("WARN", f"{TTL_VAR} 없음 — 실행 중 세션 갱신 주기가 기본값"))
    files = [settings]
    if project:
        files += [os.path.join(project, ".claude", f) for f in ("settings.json", "settings.local.json")]
    for f in files:
        if os.path.isfile(f):
            env = (load_settings(f).get("env") or {})
            for v in KEY_VARS:
                if env.get(v):
                    res.append(("FAIL", f"{f} env에 {v} — helper보다 우선. 지우세요 (`ccsw install --migrate`)"))
    for v in KEY_VARS:
        if os.environ.get(v):
            res.append(("FAIL", f"현재 셸 환경에 {v} — helper보다 우선. 환경변수를 지우고 새 창에서 실행"))
    for w in _windows_env():
        res.append(("FAIL", f"{w} — helper보다 우선. 지운 뒤 VS Code·터미널을 모두 다시 열기"))
    for p in _vscode_settings():
        if any(v in open(p, encoding="utf-8", errors="replace").read() for v in KEY_VARS):
            res.append(("FAIL", f"VS Code 설정에 키 변수 ({p}, claudeCode.environmentVariables 등) — 지우세요"))
    n = active_name()
    if not n or not read_key(n):
        res.append(("FAIL", "활성 키 없음 — `ccsw add` / `ccsw use`"))
    else:
        res.append(("OK", f"활성 키: {n} ({mask(read_key(n))})"))
    if helper:
        try:
            r = subprocess.run(helper, shell=True, capture_output=True, text=True, timeout=20)
            ok = r.returncode == 0 and r.stdout.strip() and r.stdout.strip() == (read_key(n) if n else None)
            res.append(("OK", "helper 실행 결과 = 활성 키") if ok else
                       ("FAIL", f"helper 실행 실패 (종료 코드 {r.returncode}): {r.stderr.strip()[:200]}"))
        except Exception as e:
            res.append(("FAIL", f"helper 실행 오류: {type(e).__name__}"))
    return res


def cmd_doctor(a):
    res = doctor(settings_path(a.settings), a.project)
    for st, msg in res:
        print(f"[{st:4s}] {msg}")
    bad = sum(st == "FAIL" for st, _ in res)
    print("결과: " + ("문제 없음 — Claude Code에서 /status로 인증 방식이 apiKeyHelper인지 확인" if not bad else f"FAIL {bad}건"))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ccsw", description="Claude Code API 키 이름 전환 (apiKeyHelper)")
    ap.add_argument("--settings", help="Claude Code settings.json 경로 (기본: ~/.claude/settings.json)")
    ap.add_argument("--home", help="키 저장소 폴더 (기본: ~/.claude/ccsw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install"); p.add_argument("--migrate"); p.add_argument("--ttl", type=int, default=60000)
    p.add_argument("--python", help="helper가 쓸 python 경로 (기본: 지금 실행한 python)"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("add"); p.add_argument("name"); p.add_argument("--stdin", action="store_true")
    p.add_argument("--from-env"); p.add_argument("--clipboard", action="store_true", help="클립보드에서 바로 읽기 (붙여넣기 불필요)")
    p.add_argument("--file", help="키 한 줄이 든 텍스트 파일에서 읽기 (등록 후 파일은 직접 삭제)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("use"); p.add_argument("name"); p.add_argument("--no-nudge", action="store_true")
    sub.add_parser("list"); sub.add_parser("current"); sub.add_parser("get"); sub.add_parser("uninstall")
    p = sub.add_parser("remove"); p.add_argument("name")
    p = sub.add_parser("doctor"); p.add_argument("--project", help="프로젝트 폴더 (.claude/settings*.json도 점검)")
    a = ap.parse_args(argv)
    if a.home:
        os.environ["CCSW_HOME"] = a.home
    fn = {"install": cmd_install, "add": cmd_add, "use": cmd_use, "list": cmd_list, "current": cmd_current,
          "get": cmd_get, "remove": cmd_remove, "doctor": cmd_doctor, "uninstall": cmd_uninstall}[a.cmd]
    return fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
