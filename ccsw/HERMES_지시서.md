# Hermes 작업 지시서 — ccsw 설치·검증 (Claude Code API 키 이름 전환)

## 목표
사용자의 Windows PC에서 Claude Code API 키를 `ccsw use <이름>` 한 번으로 바꾸고, **이미 실행 중인** Claude Code
(터미널·VS Code 사이드바)까지 재시작 없이 새 키를 쓰게 만든다. 도구 설명은 같은 폴더의 `README.md`.

## 절대 규칙 (어기면 즉시 중단하고 사용자에게 보고)
1. **키 값을 화면·로그·보고서·채팅에 출력하지 않는다.** `ccsw.py get`을 직접 실행하지 않는다 (doctor가 대신 확인).
   settings.json·환경변수를 보여 줄 때도 키 값은 `sk-ant-…끝4자리`로 가린다.
2. **키를 직접 입력·생성하지 않는다.** 새 키 등록은 사용자가 `ccsw.py add <이름>` 프롬프트에 직접 입력한다.
   예외: 이미 PC에 있는 키를 옮길 때(`install --migrate`, `add --from-env`)는 값을 보지 않고 옮긴다.
3. **지우기 전에 확인받는다.** Windows 환경변수 삭제, VS Code 설정 수정, VS Code·터미널 재시작은 사용자 승인 후에만.
4. 키 파일·`~/.claude/ccsw/` 폴더를 git에 넣거나 어디로도 업로드하지 않는다.
5. 의존성 설치(pip 등)·관리자 권한 명령은 사용자 승인 후에만. 시스템 환경변수(HKLM) 변경은 사용자가 직접 하게 안내만.

## 0. 준비 확인
PowerShell에서:
```powershell
python --version          # 3.8 이상. 'python'이 Microsoft Store 창을 띄우면 py -3 --version 으로 확인하고 이후 python 대신 py -3
git --version
claude --version          # Claude Code 설치 확인
```

## 1. 코드 받기
```powershell
git clone --branch claude/new-project-git-setup-hv1i9e --single-branch https://github.com/unitedrk10-sudo/day17 $env:USERPROFILE\ccsw-src
cd $env:USERPROFILE\ccsw-src\ccsw
```
(이미 받았다면 `git pull`.) 이 폴더의 `ccsw.py`, `README.md`, `tests\`를 확인.

## 2. 테스트 (선택 — pytest가 있을 때만)
```powershell
python -m pytest -q tests      # 4 passed 기대. pytest 없으면 설치 여부를 사용자에게 묻고, 거절하면 건너뜀
```

## 3. 현재 상태 점검 — 키가 지금 어디 있나
```powershell
python ccsw.py doctor
```
- 이 단계에선 `apiKeyHelper 없음` FAIL이 정상. 나머지 FAIL 줄이 **키가 들어 있는 곳**이다:
  `settings.json env` / 현재 셸 / 사용자·시스템 환경변수 / VS Code 설정.
- 프로젝트 폴더에도 Claude Code 설정이 있으면 `python ccsw.py doctor --project <프로젝트 폴더>`도 실행.
- 결과(키 값 없이)를 사용자에게 보여 주고, 기존 키에 붙일 이름(예: `work`)을 물어본다.

## 4. 설치
경우에 따라 하나:
- **(가) `~/.claude/settings.json`의 env에 키가 있음** →
  ```powershell
  python ccsw.py install --migrate <이름>
  ```
- **(나) Windows 환경변수에만 키가 있음** (현재 셸에서 `$env:ANTHROPIC_API_KEY`가 보임) →
  ```powershell
  python ccsw.py add <이름> --from-env ANTHROPIC_API_KEY      # 게이트웨이 토큰이면 ANTHROPIC_AUTH_TOKEN
  python ccsw.py install
  ```
- **(다) 어디에도 없음 / 새 키** → `python ccsw.py install` 후 사용자에게 `python ccsw.py add <이름>` 직접 입력 요청.

추가 키가 있으면 사용자가 `python $env:USERPROFILE\.claude\ccsw\ccsw.py add <이름>`으로 직접 등록.
설치하면 `settings.json.bak-<시각>` 백업이 생긴다 — 경로를 기록해 둔다.

## 5. helper를 가리는 키 변수 제거 (사용자 승인 후)
`doctor`가 FAIL로 알려준 곳만:
- **사용자 환경변수**: 승인 후
  ```powershell
  [Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $null, "User")
  [Environment]::SetEnvironmentVariable("ANTHROPIC_AUTH_TOKEN", $null, "User")
  ```
- **시스템 환경변수**: 관리자 권한 필요 — 사용자에게 직접 지우도록 안내 (시스템 속성 → 환경 변수).
- **VS Code 설정** (`%APPDATA%\Code\User\settings.json`의 `claudeCode.environmentVariables` 등): 해당 항목만 지우는 편집을 사용자 승인 후. 파일 백업 먼저.
- **프로젝트 `.claude\settings*.json` env**: 해당 키 줄만 삭제 (승인 후, 백업 먼저).

## 6. 한 번만 재시작 (사용자에게 요청)
"Claude Code가 떠 있는 터미널과 VS Code 창을 **모두** 닫았다가 다시 열어 주세요. 이번 한 번뿐이고 이후 키 전환은 재시작이 필요 없습니다."
새 PowerShell 창에서:
```powershell
python $env:USERPROFILE\.claude\ccsw\ccsw.py doctor      # FAIL 0건이어야 함
```

## 7. 동작 확인 (사용자와 함께)
1. 터미널의 Claude Code와 VS Code 사이드바에서 각각 `/status` → 인증 방식이 **apiKeyHelper**인지 사용자에게 확인받는다.
2. 실시간 전환 시험:
   ```powershell
   "sk-ant-invalid-ccsw-test" | python $env:USERPROFILE\.claude\ccsw\ccsw.py add bogus --stdin
   python $env:USERPROFILE\.claude\ccsw\ccsw.py use bogus
   ```
   70초 기다린 뒤, **재시작하지 않은** 기존 세션(터미널·VS Code 둘 다)에서 아무 질문 → 인증 오류가 나면 성공.
   ```powershell
   python $env:USERPROFILE\.claude\ccsw\ccsw.py use <원래 이름>
   ```
   70초 뒤 같은 세션에서 다시 질문 → 정상 응답이면 성공.
   ```powershell
   python $env:USERPROFILE\.claude\ccsw\ccsw.py remove bogus
   ```
3. (선택) `%USERPROFILE%\.claude\ccsw`를 사용자 PATH에 추가하면 `ccsw use <이름>`으로 짧게 쓸 수 있다 — 사용자 승인 후.

## 8. 되돌리기 (문제가 생기면)
```powershell
python $env:USERPROFILE\.claude\ccsw\ccsw.py uninstall
```
그래도 이상하면 4단계에서 기록한 `settings.json.bak-<시각>`을 `settings.json`으로 복원하고, 세션을 다시 연다.
키 파일(`%USERPROFILE%\.claude\ccsw\keys\`)은 지우지 않는다 (사용자가 원할 때만).

## 9. 보고 (키 값 없이)
```
- 설치 경우: (가)/(나)/(다) · 등록한 키 이름: ...
- 지운 키 변수 위치: ...
- doctor 결과: FAIL 0건 (또는 남은 FAIL 줄)
- /status 인증 방식: 터미널 ___ · VS Code ___
- 실시간 전환 시험: 틀린 키로 오류 확인 O/X · 원래 키 복귀 O/X (TTL 몇 초 뒤)
- settings 백업 경로: ...
- 문제·특이사항: ...
```
