# ccsw — Claude Code API 키 이름 전환

`ccsw use work` 한 번으로 **이미 실행 중인** Claude Code 세션(터미널, VS Code 사이드바, `-p` 스크립트)까지 키가 바뀐다.
표준 라이브러리만 쓰는 Python 파일 하나 (`ccsw.py`, Python 3.8+). Windows 우선, macOS·Linux 동작.

## 왜 환경변수로는 안 되나
- 환경변수(`ANTHROPIC_API_KEY`)는 프로세스가 **시작할 때 복사**된다 → 이미 떠 있는 Claude Code·VS Code는 옛 값을 계속 쓴다.
- Claude Code 인증 우선순위: `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` → **`apiKeyHelper`** → 로그인.
  위 두 환경변수가 하나라도 남아 있으면 helper는 무시된다.

## 원리
`~/.claude/settings.json`의 `apiKeyHelper`가 `ccsw.py get`을 실행해 키를 받는다. Claude Code는
`CLAUDE_CODE_API_KEY_HELPER_TTL_MS`(기본 설치값 60초)마다 다시 받고, settings.json이 바뀌면 실행 중인 세션에 다시 적용한다.
`ccsw use`는 활성 키를 바꾸고 settings.json을 같은 내용으로 다시 써서 반영을 앞당긴다 → **늦어도 TTL 안에 모든 세션이 새 키**.

## 설치 (한 번)
```
python ccsw.py doctor                         # 지금 키가 어디 있는지 점검 (키 값은 출력 안 함)
python ccsw.py install --migrate work         # settings.json env의 기존 키를 'work'로 옮기고 helper 설정 (백업 생성)
#   env에 키가 없으면:  python ccsw.py install  →  python ccsw.py add work   (화면에 안 보이게 입력)
```
그다음 **딱 한 번** 터미널·VS Code 창을 모두 닫았다 다시 연다 (떠 있던 세션은 예전 env 키를 기억함).
Windows 사용자/시스템 환경변수나 VS Code 설정(`claudeCode.environmentVariables`)에 키가 있으면 지운다 — `doctor`가 알려준다.

## 사용
```
python %USERPROFILE%\.claude\ccsw\ccsw.py add personal      # 키 추가
python %USERPROFILE%\.claude\ccsw\ccsw.py use personal      # 전환 (재시작 불필요)
python %USERPROFILE%\.claude\ccsw\ccsw.py list              # * = 활성, 키는 sk-ant-…ab12처럼 가림
python %USERPROFILE%\.claude\ccsw\ccsw.py doctor
```
Windows에서는 설치 때 `%USERPROFILE%\.claude\ccsw\ccsw.cmd`도 만들어진다 — 그 폴더를 PATH에 넣으면 `ccsw use personal`.

| 명령 | 하는 일 |
|---|---|
| `install [--migrate 이름] [--ttl ms] [--python 경로]` | helper 설정 (settings.json 백업 후), 스크립트를 `~/.claude/ccsw/`로 복사 |
| `add 이름 [--stdin \| --from-env 변수] [--force]` | 키 등록 (`keys/이름.key`, 본인만 읽기 권한) |
| `use 이름 [--no-nudge]` | 활성 키 전환 |
| `list` / `current` | 목록 / 현재 키 (가려서) |
| `remove 이름` | 삭제 (활성 키는 불가) |
| `doctor [--project 폴더]` | helper를 무시하게 만드는 키 변수(셸·Windows 환경변수·settings env·VS Code 설정) 점검, helper 실행 확인 |
| `uninstall` | settings.json에서 helper·TTL 제거 (키 파일은 남김) |
| `get` | helper 전용 — 활성 키 원문 출력. 사람이 실행하지 말 것 |

## 확인
- Claude Code에서 `/status` → 인증 방식이 `apiKeyHelper`
- 실시간 전환 시험: 일부러 틀린 키로 `use` → TTL 지나서 기존 세션 요청이 인증 오류 → 원래 키로 `use` → 정상

## 한계
- Claude Desktop 앱·클라우드 세션은 apiKeyHelper를 쓰지 않는다 (CLI·VS Code/JetBrains 확장·Agent SDK·`-p`만).
- 구독 로그인(`/login`) 계정 전환은 대상 아님.
- `CLAUDE_CONFIG_DIR`로 설정 폴더를 나눠 쓰면 폴더마다 `install` (`--settings`로 지정).
- 설정 재작성으로 helper 캐시가 즉시 비워지는지는 문서에 명시돼 있지 않다 — 보장되는 것은 TTL 주기 갱신.

## 보안
- 키는 `~/.claude/ccsw/keys/*.key` (Windows: icacls로 본인만, 그 외: 0600). 화면·로그에는 가린 값만.
- 이 폴더·키 파일을 리포에 넣지 말 것. `get`은 helper만 실행.

## 테스트
`python -m pytest -q tests` — 임시 폴더만 사용 (실제 `~/.claude`·환경변수·레지스트리 안 건드림).
