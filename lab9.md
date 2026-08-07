# Lab 9. Code Interpreter를 이용한 데이터 시각화와 Skill

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"Claude Code Skills"(p.72)**, **"Skill 구성요소"(p.73)**, **"SKILL.md"(p.74)** 이론을 `[실습]_9_Code_Interpreter를_이용한_데이터_시각화와_Skill.ipynb`의 실제 실행 결과와 결합하여 재구성한 것입니다. README 부록 "실습(Lab) 목록"의 **Lab #10 "LangChain Code Interpreter를 이용한 데이터 시각화"(p.75, Day2-3)**에 해당합니다.
>
> - README 소개: "LangChain과 Python Code Generator 연동하기 / 데이터 시각화 코드 작성과 개선, 실행 기능 연동하기 / 시각화 결과 평가 및 개선 모듈 연결하기 / **Skill 개념과 실행 과정 이해하기**"

## 학습 목표

- 여러 개의 개별 Tool을 일일이 연결하는 대신, **CLI 셸 명령 실행 도구(`run_command`) 하나로 파이썬 코드 실행·데이터 시각화까지 처리**하는 "Code Interpreter" 패턴을 구현한다.
- 셸 도구가 강력한 만큼 위험하므로, **비밀 파일 차단·위험 명령 차단·작업 디렉터리 고정**이라는 3가지 안전장치를 실제로 구현하고, 문자열 패턴 검사의 한계(우회 가능성, 정상 명령의 오탐)를 실제로 확인한다.
- **Skill**의 핵심 메커니즘인 **"점진적 공개(progressive disclosure)"**를 `SKILL.md`의 프론트매터/본문 분리 구조로 직접 구현하고, 여러 Tool을 미리 다 붙이는 대신 "필요할 때 로드"하는 방식이 컨텍스트 문제를 어떻게 해결하는지 확인한다.
- 매출/재고 SQL 스킬과 보고서 생성 스킬을 조합하여, 하나의 자연어 요청으로 **여러 스킬을 순차적으로 로드·활용**하는 멀티스킬 에이전트를 실행해본다.
- [실습 과제]로 새로운 Skill(`clock`)을 직접 만들고 테스트하는 전체 과정(SKILL.md 작성 → 스크립트 작성 → 실행 테스트 → 평가 데이터셋 작성)을 실제로 수행한다.

## Part 0. 개요와 노트북 구조

**노트북 42개 셀의 흐름**:
1. 환경설정 (패키지 설치, LLM 준비)
2. **CLI 도구 연결하기** — 안전 경계가 있는 셸 실행 도구(`run_command`) 구현
3. 셸 도구만 가진 에이전트로 **데이터 시각화**(제품별 판매량 → `result.png`) 수행
4. **Skill(스킬)** 개념 소개, `SKILL.md` 구조, 5개 예시 스킬 스캔
5. `load_skill` 도구 구현 — 점진적 공개의 실제 구현
6. `read_file`/`write_file`/`list_files` 추가, 멀티스킬 에이전트 구성
7. 실전 시연: 매출 분석(단일 스킬), 매출+재고 종합 보고서(다중 스킬)
8. **[실습] 새로운 Skill 만들기** — `clock` 스킬을 직접 작성하고 테스트

---

## Part 1. 환경 준비

### 이론: LangChain 실용 예제 개요 (p.60)

> LangChain의 부가적인 기능을 활용하여 다양한 작업 가능 — Tool과 Agent 만들기: **Tool**: LLM이 참고할 수 있는 다양한 외부 기능(함수, API 등)

이번 실습에서는 지금까지의 실습(Lab 8의 여러 개별 Tool)과 다르게, **터미널 명령을 실행하는 단 하나의 강력한 Tool**을 중심으로 에이전트를 구성합니다.

```python
!pip install openai matplotlib langchain langchain-google-genai langchain-openai langchain-community langchain-experimental langchain-tavily langgraph -q
```

**실행 결과**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-colab 1.0.0 requires requests==2.32.4, but you have requests 2.34.2 which is incompatible.
```

```python
# 한국어 시각화 라이브러리
!pip install koreanize_matplotlib -q
# Python 3.13 이후에는 setuptools 라이브러리를 같이 설치해주세요
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(model = 'gpt-5.2', reasoning_effort='low')
```

> 💡 `koreanize_matplotlib`은 matplotlib의 한글 폰트/마이너스 기호 깨짐을 자동으로 고쳐주는 라이브러리입니다 — 이후 에이전트가 생성하는 시각화 스크립트에서 반복적으로 `import koreanize_matplotlib`을 사용하도록 시스템 프롬프트에 명시됩니다.

---

## Part 2. CLI 도구 연결하기 — 안전한 셸 실행 도구 만들기

> 최근 에이전트의 흐름은 여러 개의 도구를 연결하는 대신, 터미널 CLI 도구를 연결하고 예시 스크립트를 전달하는 방법으로 변화하고 있습니다.

### 이론(노트북 원문): 셸 도구의 안전 경계

> 셸 도구는 임의의 명령을 실행할 수 있으므로, 다른 도구에 걸어 둔 제약을 우회하는 통로가 됩니다. 파일 읽기 도구에 차단 목록을 지정하더라도, 셸이 열려 있으면 같은 파일을 터미널 명령으로 읽을 수 있습니다.
>
> 따라서 셸 도구에는 최소한 다음 세 가지를 적용합니다.
> - **비밀 정보 파일 차단**: 명령어에 `.env`, `credentials`, `id_rsa` 등이 포함되면 실행하지 않습니다.
> - **위험 명령 차단**: 재귀 삭제, 디스크 포맷, 권한 상승, 내려받은 스크립트의 즉시 실행을 막습니다.
> - **실행 위치 고정**: `cwd`를 지정해 작업 디렉터리 안에서만 명령이 실행되도록 합니다.
>
> 문자열 패턴 검사는 표현을 바꾸면 통과할 수 있어, 이것만으로 완전한 격리가 되지는 않습니다. 운영 환경에서는 컨테이너, 별도 실행 계정, 읽기 전용 마운트처럼 운영체제 수준의 격리를 함께 사용합니다.

> 💡 **README와의 연결**: 이 경고는 README p.61 "SQL in LangChain"의 "쿼리 실행의 경우 추가적 작업이 필요하며, 안전 문제를 고려해야 함"과 Lab 7의 Human-in-the-Loop 안전장치와 **정확히 같은 원리를 셸 명령 전체로 일반화**한 것입니다 — SQL 쿼리든 셸 명령이든, "임의 실행 가능한 인터페이스를 LLM에 주는 것"은 항상 이런 3중 안전장치(민감 대상 차단 / 위험 동작 차단 / 실행 범위 제한)가 필요합니다.

### 실습: 안전 검사가 포함된 `run_command` 도구

```python
from langchain_core.tools import tool
from pathlib import Path
import os, re, subprocess

WORKSPACE = Path.cwd().resolve()

# 명령어 문자열에 등장하면 실행을 막을 비밀 정보 파일
BLOCKED_FILES = ['.env', 'credentials', '.secret', 'id_rsa', '.pem']

# 되돌릴 수 없거나 권한을 넘기는 명령 패턴
BLOCKED_PATTERNS = [
    r'\brm\s+-[a-z]*[rf]',          # rm -rf
    r'\b(del|rmdir)\s+/[sqf]',      # Windows 재귀 삭제
    r'\b(format|mkfs|fdisk)\b',
    r'\b(shutdown|reboot)\b',
    r'\b(sudo|su)\s',
    r'(^|[\s/\\])env(\s|$)',        # env 파일 직접 참조
    r'>\s*/dev/[sh]d',              # 디스크에 직접 쓰기
    r'\b(curl|wget)\b[^|]*\|\s*(ba)?sh',   # 내려받은 스크립트 즉시 실행
]


def check_command(command: str) -> str | None:
    """차단 사유를 반환합니다. 실행해도 되는 명령이면 None을 반환합니다."""
    lowered = command.lower()
    for name in BLOCKED_FILES:
        if name in lowered:
            return f"비밀 정보 파일({name})에 접근하는 명령입니다."
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return f"위험한 명령 패턴({pattern})이 포함되어 있습니다."
    return None


@tool
def run_command(command: str, timeout:int=60) -> str:
    """터미널 명령어를 실행합니다. command: 실행할 셸 명령어"""
    reason = check_command(command)
    if reason:
        return f'[차단] {reason}'
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=WORKSPACE)
        out = r.stdout + (f"\n[STDERR] {r.stderr}" if r.returncode != 0 else '')
        return out.strip() or '(완료, 출력 없음)'
    except Exception as e:
        return f'오류: {e}'
```

> 💡 `check_command()`가 3중 안전장치 중 **비밀 파일 차단**과 **위험 명령 차단**을 구현하고, `subprocess.run(..., cwd=WORKSPACE)`가 **실행 위치 고정**을 담당합니다. `BLOCKED_PATTERNS`의 정규식들은 `rm -rf`, Windows `del /s`, `format`/`mkfs`/`fdisk`(디스크 포맷), `shutdown`/`reboot`, `sudo`/`su`(권한 상승), `curl|wget ... | sh`(다운로드 즉시 실행) 등 **되돌릴 수 없는 파괴적 명령**을 폭넓게 차단합니다. (이 목록의 `\b(format|mkfs|fdisk)\b` 패턴이 Part 9에서 실제로 예상치 못한 오탐을 일으키는데, 뒤에서 그 사례를 다룹니다.)

### 실습: 기본 동작 확인

```python
# 현재 폴더 검색
run_command.invoke('ls')
```

**실행 결과**: `'sample_data\nsample_data.csv\nskills.zip'`

```python
# 임의의 파이썬  스크립트 실행
run_command.invoke({"command": 'python -c "print(1+2)"'})
```

**실행 결과**: `'3'`

### 실습: 예시 데이터 확인

```python
import pandas as pd
df = pd.read_csv('./sample_data.csv')
df.head()
```

**실행 결과**:
```
           날짜   제품       가격  판매량  고객_연령 고객_성별  지역
0  2023-01-01  노트북  1200000    5     34    남성  서울
1  2023-01-02  노트북  1200000    3     42    여성  부산
2  2023-01-03  노트북  1200000    2     28    남성  대구
3  2023-01-04  노트북  1200000    7     45    여성  인천
4  2023-01-05  노트북  1200000    4     31    남성  광주
```

---

## Part 3. 셸 도구를 가진 에이전트로 데이터 시각화하기

> 정의한 셸 도구를 LangChain Agent에 연결합니다. 체크포인터로 `InMemorySaver`를 전달하면 `thread_id` 단위로 대화가 이어집니다.

### 이론: Agent — 툴을 이용한 복잡한 작업 수행 (p.69)

> Tool의 결과물을 얻는 것에서 끝나지 않고, 해당 Tool을 사용해 문제 해결하기 — LLM → 함수 실행 → 함수 결과 → LLM에 다시 전달 → 최종 답변 생성

### 실습: 에이전트 구성

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

tools = [run_command]

system_prompt = '''당신은 파이썬 코드를 이용한 데이터 분석의 전문가입니다.
툴을 사용하기 전에는 중간 단계를 간략하게 설명하세요.
한국어 시각화를 다루는 경우, koreanize_matplotlib을 항상 import하세요'''

agent = create_agent(llm, tools=tools, system_prompt=system_prompt, checkpointer=InMemorySaver())

config = {'configurable': {'thread_id': 'cli-agent'}}

agent
```

**실행 결과**: `<langgraph.graph.state.CompiledStateGraph object at 0x7a2a646f6810>`

### 실습: "제품별 판매량을 시각화해줘" — 전체 실행 트레이스

```python
question = "파일의 위치가 ./sample_data.csv 이다. 이 파일의 제품별 판매량을 비교하여 result.png에 시각화해줘."

# 깔끔하게 스트리밍
for chunk in agent.stream({'messages':[HumanMessage(question)]}, config=config, stream_mode="updates"):
    for step, data in chunk.items():
        print(f"step: {step}")
        result = data['messages'][-1].text
        if len(result) > 1000:
            print(f"    content: {result[0:200] } (중략)")
        else:
            print(f"    content: {result}")

        if step=='model':
            print("T.C:", data['messages'][-1].tool_calls)
        print('-------------')
```

**실행 결과** (에이전트가 스스로 계획 → 실행 → 재계획 → 실행 → 최종 답변한 4단계):

**1단계 — 계획**: "먼저 (1) CSV의 컬럼 구성을 확인한 뒤, (2) '제품'에 해당하는 컬럼으로 판매량을 집계(groupby)하고, (3) 막대그래프로 저장(result.png)하겠습니다." → `run_command(command="python - <<'PY' ... df.columns ... PY")` 호출

**2단계 — 컬럼 확인 결과**:
```
날짜   제품       가격  판매량  고객_연령 고객_성별  지역
Index(['날짜', '제품', '가격', '판매량', '고객_연령', '고객_성별', '지역'], dtype='object')
```

**3단계 — 시각화 스크립트 실행**: "컬럼에 '제품'과 '판매량'이 있으니, 제품별로 판매량 합계를 계산한 뒤 내림차순 막대그래프를 그리고 한국어 폰트 처리를 위해 `koreanize_matplotlib`을 import하여 `result.png`로 저장하겠습니다."

```python
run_command(command="""python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한국어 폰트/마이너스 깨짐 방지

df = pd.read_csv('./sample_data.csv')
sales_by_product = df.groupby('제품', as_index=True)['판매량'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sales_by_product.plot(kind='bar', color='#4C78A8')
ax.set_title('제품별 판매량 비교')
ax.set_xlabel('제품'); ax.set_ylabel('판매량(합계)')
ax.grid(axis='y', linestyle='--', alpha=0.4)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f'{int(h)}', (p.get_x() + p.get_width()/2, h),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig('result.png', dpi=200)
print('Saved: result.png')
print(sales_by_product)
PY""")
```

**4단계 — 실행 결과**:
```
Saved: result.png
제품
이어폰      113
스마트워치     63
스마트폰      55
태블릿       35
노트북       21
Name: 판매량, dtype: int64
```

**최종 답변**: "먼저 CSV의 컬럼을 확인한 뒤, `제품`별로 `판매량`을 합계 집계해서 막대그래프로 저장했습니다. (한국어 표시를 위해 `koreanize_matplotlib`을 import) 결과 파일: `result.png` / 제품별 판매량(합계): 이어폰 113, 스마트워치 63, 스마트폰 55, 태블릿 35, 노트북 21"

> 💡 **주목할 점**: 에이전트는 사용자가 시각화만 요청했음에도, **먼저 데이터 구조를 스스로 탐색**(컬럼 확인) → **탐색 결과에 맞춰 시각화 코드를 작성** → **실행 결과를 검증하고 요약**하는 3단계를 자율적으로 수행했습니다. 이는 README p.69의 "Planning, Tool Executing, Observation을 포함하여 문제 해결"이 실제 코드 실행으로 구현된 사례입니다 — Lab 8의 "고정된 Tool 하나(계산기 등)"와 달리, 이 실습에서는 **"파이썬 코드를 실행하는 셸" 하나가 곧 무한히 유연한 Tool**이 됩니다(Code Interpreter 패턴의 핵심).

---

## Part 4. Skill(스킬)

> LLM에 여러 개의 툴을 연결하는 경우, 컨텍스트가 길어지는 문제가 발생합니다. 이를 해결하기 위한 방법으로, Skill(스킬)이 주목받고 있습니다.
>
> 스킬의 핵심 메커니즘은 **'점진적 공개'**입니다. 복잡한 작업을 수행해야 하거나, 연결된 도구가 많아지면 컨텍스트가 증가하는 문제를 해결합니다.

### 이론: Claude Code Skills (p.72) + Skill 구성요소 (p.73) + SKILL.md (p.74)

> - Tool 등의 추가 Context가 너무 많아지는 경우, 해당 내용을 불러오는 Overhead가 너무 많아지는 **Context Bloat** 발생
> - Context를 **Foldable 구조**로 구조화하여, 필요한 내용만 불러오는 Skill 제안 — 최초에는 짧은 설명만을 전달하며, 이후 에이전트가 필요한 스킬을 로드하여 컨텍스트에 저장하는 방법
> - Tool / MCP와 다르게, 형식이 정해져 있지 않은 폴더(Folder) 구조 — `SKILL.md`(스킬 사용 설명), 프롬프트, Tool 역할의 스크립트, 예시 문서
> - **YAML Frontmatter**와 점진적 공개 메커니즘 — Frontmatter: 이름과 짧은 소개를 사전에 전달. 이후 해당 스킬을 로드(read)하면, 뒷부분의 내용을 컨텍스트로 가져와 활용

### 실습: `SKILL.md`의 형식 (노트북 원문)

스킬은 `SKILL.md` 파일로 정의하며, YAML frontmatter(메타데이터)와 Markdown body(지침)로 구성됩니다.

```yaml
---
name: skill-name              # 스킬 식별자 (소문자, 하이픈)
description: 한 줄 설명         # 에이전트의 자동 호출 판단에 사용
metadata:
  type: cli-tool               # cli-tool | prompt-bundle | file-template
---

Markdown 본문: DB 스키마, 비즈니스 로직, 스크립트 사용법 등
```

### 실습: 스킬 압축 해제와 폴더 구조

```python
import zipfile
import os

def unzip_file(zip_file_path, extract_path='./'):
    """
    zip 파일을 지정된 경로에 압축 해제합니다.
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"'{zip_file_path}' 파일이 '{extract_path}'에 성공적으로 압축 해제되었습니다.")
    except zipfile.BadZipFile:
        print(f"오류: '{zip_file_path}'은 유효한 zip 파일이 아닙니다.")
    except FileNotFoundError:
        print(f"오류: '{zip_file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"압축 해제 중 오류 발생: {e}")

unzip_file('skills.zip')
```

**실행 결과**: `'skills.zip' 파일이 './'에 성공적으로 압축 해제되었습니다.`

**스킬 폴더 구조** (노트북 원문):
```
skills/
├── sales_analytics/           ← CLI Tool: 매출 DB 스키마 + 쿼리 스크립트
│   ├── SKILL.md               (YAML frontmatter + 테이블 구조, 비즈니스 로직)
│   ├── setup.py               (SQLite 샘플 데이터 생성)
│   └── query.py               (SQL 쿼리 실행 유틸리티)
├── inventory_management/      ← CLI Tool: 재고 DB 스키마 + 쿼리 스크립트
│   ├── SKILL.md
│   ├── setup.py
│   └── query.py
└── customer_report/           ← Prompt Bundle + File Template
    ├── SKILL.md               (보고서 작성 가이드라인, 스타일 규칙)
    └── generate.py            (매출/재고 DB 연계 보고서 생성)

(+ 2개의 추가 스킬)
```

### 실습: 스킬 스캐너 — 프론트매터만 가볍게 파싱하기

> 위 내용 중, 프론트매터에 해당하는 부분만 최초 에이전트에 연결됩니다. 이후, 에이전트가 해당 스킬이 필요할 때 스킬의 전체 내용을 로드하는 방식입니다.

```python
import yaml
from pathlib import Path

SKILLS_DIR = Path("skills")


def scan_skills(skills_dir: Path = SKILLS_DIR) -> dict[str, dict]:
    """SKILL.md의 YAML frontmatter를 파싱하여 {name: {description, path, meta, body}} 딕셔너리를 반환합니다."""
    skills = {}
    if not skills_dir.exists():
        return skills
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir():
            continue
        skill_file = d / "SKILL.md"
        if not skill_file.exists():
            continue
        content = skill_file.read_text(encoding="utf-8")
        # --- frontmatter --- 파싱
        if not content.startswith("---"):
            continue
        parts = content.split("---", 2)
        if len(parts) < 3:
            continue
        meta = yaml.safe_load(parts[1])
        body = parts[2].strip()
        name = meta.get("name", d.name)
        skills[name] = {
            "name": name,
            "description": meta.get("description", ""),
            "path": str(d),
            "folder": d.name,
            "meta": meta,
            "body": body,
        }
    return skills


def format_skill_list(skills: dict[str, dict]) -> str:
    """스킬 목록을 이름, 타입, 설명 한 줄씩으로 정리합니다."""
    lines = [f"사용 가능한 스킬 {len(skills)}개:"]
    for name, s in skills.items():
        skill_type = s["meta"].get("metadata", {}).get("type", "unknown")
        lines.append(f"  - {name} [{skill_type}]: {s['description']}")
    return "\n".join(lines)


print(format_skill_list(scan_skills()))
```

**실행 결과**:
```
사용 가능한 스킬 5개:
  - customer-report [prompt-bundle]: 고객 분석 보고서 생성 가이드라인과 마크다운 템플릿 (매출 DB + 재고 DB 연계 분석)
  - inventory-management [cli-tool]: 재고 관리를 위한 DB 스키마와 SQL 쿼리 스크립트 (products, warehouses, inventory, stock_movements 테이블)
  - sales-analytics [cli-tool]: 매출 데이터 분석을 위한 DB 스키마와 SQL 쿼리 스크립트 (customers, orders, order_items 테이블)
  - skill-creator [unknown]: Create new skills, modify and improve existing skills, and measure skill performance. ...
  - web-search [cli-tool]: Tavily 웹 검색(카테고리/시간범위/도메인 필터/원본본문 등 전체 옵션)과 URL 본문 추출(fetch_url) 스크립트. 최신 정보 조회, 외부 문서 정독, 실시간 뉴스/금융/리서치에 사용
```

> 💡 **점진적 공개의 1단계가 바로 이 출력**입니다 — 시스템 프롬프트에는 **이 5줄(스킬 이름+타입+한 줄 설명)만** 포함되고, 각 스킬의 상세 내용(DB 테이블 스키마, SQL 예시, 보고서 규칙 등 수백~수천 자)은 아직 컨텍스트에 없습니다. README p.72의 "Context Bloat"를 방지하는 핵심 아이디어가 바로 이것입니다.

---

## Part 5. `load_skill` 도구 — 점진적 공개의 실제 구현

> 스킬을 불러오기 위한 기본 도구를 구성합니다. `skill_name`을 통해 스킬의 상세 내용에 접근하고, 이를 컨텍스트에 추가합니다. `load_skill`은 호출될 때마다 `skills/`를 다시 읽으므로, 스킬을 추가하면 도구는 곧바로 인식합니다.

```python
@tool
def load_skill(skill_name: str) -> str:
    """스킬의 상세 내용(SKILL.md body)을 로드합니다. 해당 도메인의 작업을 수행하기 전에 호출하세요.

    Args:
        skill_name: 로드할 스킬 이름
    """
    skills = scan_skills()      # 호출 시점의 skills/ 상태를 그대로 읽습니다
    if skill_name not in skills:
        return f"스킬 '{skill_name}'을 찾을 수 없습니다. 사용 가능: {', '.join(skills)}"

    skill = skills[skill_name]
    body = skill["body"]

    # 스킬 폴더의 파이썬 스크립트를 실행 명령과 함께 덧붙입니다.
    scripts = [f.name for f in Path(skill["path"]).iterdir() if f.is_file() and f.suffix == ".py"]
    if scripts:
        body += "\n\n## 실행 가능한 스크립트\n"
        for s in scripts:
            body += f"- `{skill['folder']}/{s}` → `python skills/{skill['folder']}/{s}`\n"

    return body
```

```python
load_skill.invoke("customer-report")
```

**실행 결과**:
```
## 보고서 구조

보고서는 반드시 아래 섹션 순서를 따릅니다:

1. 요약 (Executive Summary) -- 핵심 지표 3~5개, 한 문단
2. 매출 분석 -- 기간별 매출 추이, 지역별/등급별 분포
3. 고객 분석 -- 활성 고객 수, CLV 상위 고객, 이탈 위험 고객
4. 재고 현황 -- 재주문 필요 상품, 창고별 재고 분포
5. 권장 액션 -- 데이터 기반 구체적 제안 3~5개

## 스타일 규칙

- 수치에는 반드시 단위 표기 (원, 건, 명)
- 금액은 천 단위 쉼표 사용 (예: 1,234,000원)
- 비율은 소수점 1자리까지 (예: 23.5%)
- 차트/테이블 포함 시 캡션 필수
- 파일명: report_YYYYMMDD_주제.md

## Scripts

- generate.py -- 매출/재고 DB에서 데이터를 읽어 보고서 초안 생성
  - Usage: python generate.py --type summary
  - Options: --type summary|sales|inventory|full

## 실행 가능한 스크립트
- customer_report/generate.py → python skills/customer_report/generate.py
```

> 💡 `load_skill.invoke("customer-report")` 하나로, Part 4에서 이름과 한 줄 설명만 알던 스킬의 **전체 지침(보고서 섹션 순서, 표기 규칙, 실행 가능한 스크립트 경로)**이 한꺼번에 반환됩니다. 이 내용은 `load_skill`이 호출된 그 순간에만 대화 컨텍스트에 추가되므로, **사용하지 않는 스킬의 상세 내용은 컨텍스트를 전혀 차지하지 않습니다** — README p.74의 "점진적 공개 메커니즘"이 정확히 이렇게 동작합니다.

---

## Part 6. 파일 시스템 도구 추가와 멀티스킬 에이전트 구성

> 2개의 툴과 함께 사용할 기본 파일 시스템 도구를 추가하고, 에이전트를 구성합니다.

### 이론: Tools in LangChain (p.67), LLM Tool Calling — Taker에서 Giver로 (p.66)

> - Custom 함수의 경우에도, decorator를 통해 tool로 변환 가능
> - 외부 Tool과 Text로 소통할 수 있는 인터페이스만 주어지면 양방향 활용 가능 — 파일 시스템 접근, API 호출, ...
> - 예시: User: "정리해서 report.txt에 저장해줘." → LLM: `Function Call {Tool: Write_file, ...}` → Tool: `{"Result":"저장 완료!"}` → "리포트 저장이 완료되었습니다!"

> 💡 README p.66의 예시(`Write_file` Tool로 리포트 저장)가 이번 실습 Part 8에서 **실제로 그대로 재현**됩니다 — `write_file` 도구로 `report_20260804_매출_재고_종합.md`를 저장하는 장면이 바로 이 이론 슬라이드의 실사례입니다.

### 실습: 경로 탈출을 막는 안전한 파일 도구

```python
import os
from pathlib import Path
from langchain.tools import tool

# 이름이 일치하거나 확장자가 일치하면 접근을 막을 파일
BLOCKED_FILENAMES = ['env', '.env', 'credentials', '.secret', 'id_rsa', '.pem']


def is_blocked_name(file_path: str) -> bool:
    """파일 이름이 차단 목록에 해당하는지 확인합니다."""
    base = os.path.basename(file_path)
    return base in BLOCKED_FILENAMES or any(file_path.endswith(b) for b in BLOCKED_FILENAMES)


def safe_path(p: str) -> Path:
    """작업 디렉터리(WORKSPACE) 안의 경로로 정규화합니다.
    realpath로 상위 경로 참조와 심볼릭 링크까지 해소한 뒤 비교합니다."""
    target = Path(os.path.realpath(WORKSPACE / p))
    try:
        inside = os.path.commonpath([str(WORKSPACE), str(target)]) == str(WORKSPACE)
    except ValueError:      # 서로 다른 드라이브라 비교할 수 없는 경우
        inside = False
    if not inside:
        raise ValueError(f'작업 디렉터리 밖 경로 접근 불가: {p}')
    return target


@tool
def read_file(file_path: str) -> str:
    """파일 내용을 읽습니다. file_path: 파일 경로"""
    if is_blocked_name(file_path):
        return f"보안 정책에 의해 '{os.path.basename(file_path)}' 파일은 읽을 수 없습니다."
    try:
        text = safe_path(file_path).read_text(encoding='utf-8')
        return text[:10000] + ('\n... (truncated)' if len(text) > 10000 else '')
    except Exception as e:
        return f'파일 읽기 오류: {e}'


@tool
def write_file(file_path: str, content: str) -> str:
    """파일에 내용을 작성. file_path: 경로, content: 작성할 내용"""
    if is_blocked_name(file_path):
        return f"보안 정책에 의해 '{os.path.basename(file_path)}' 파일은 쓸 수 없습니다."
    try:
        target = safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f'파일 작성 완료: {file_path} ({len(content)}자)'
    except Exception as e:
        return f'파일 작성 오류: {e}'


@tool
def list_files(path: str = ".") -> str:
    """디렉터리의 파일/하위 폴더 목록을 반환합니다.

    Args:
        path: 작업 디렉터리 기준 상대 경로 (기본값: 현재 디렉터리)
    """
    try:
        target = safe_path(path)
        if not target.is_dir():
            return f"[ERROR] 디렉터리가 아님: {path}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"[ERROR] {e}"


tools = [load_skill, run_command, read_file, write_file, list_files]
```

> 💡 **경로 탈출 방지(Path Traversal 방어)**: `safe_path()`는 `os.path.realpath()`로 **`../` 상위 경로 참조와 심볼릭 링크까지 모두 해소**한 뒤, 결과 경로가 여전히 `WORKSPACE` 내부인지 `os.path.commonpath()`로 검증합니다. 예를 들어 `file_path='../../etc/passwd'`처럼 작업 디렉터리를 벗어나려는 시도는 `ValueError`로 즉시 차단됩니다. 이는 Part 2의 `run_command` 안전장치(문자열 패턴 매칭)와는 다른 종류의 방어(경로 정규화 기반)이며, **두 종류의 안전장치를 함께 쓰는 것**이 방어를 더 견고하게 만듦을 보여줍니다.

### 실습: 스킬 목록을 시스템 프롬프트에 주입한 에이전트

> 프론트매터 목록을 시스템 프롬프트에 추가해 에이전트를 만듭니다.

```python
agent = create_agent(
    llm, tools=tools,
    system_prompt=f"""도구를 활용할 때마다, 중간 과정을 간략히 설명하세요.
{format_skill_list(scan_skills())}""",
    checkpointer=InMemorySaver(),
)

skill_config = {'configurable': {'thread_id': 'skill-agent'}}
```

### 실습: 스트리밍 헬퍼 함수

```python
def stream_agent(agent, input_msgs, config=None):
    """에이전트 실행을 스트리밍하며 각 스텝을 출력합니다."""
    def _format_arg(v):
        s = repr(v) if not isinstance(v, str) else v
        return s if len(s) <= 200 else s[:197] + "..."

    last_msg = None
    for chunk in agent.stream(input_msgs, config=config, stream_mode="updates"):
        for step, data in chunk.items():
            msg = data['messages'][-1]
            content = msg.text if hasattr(msg, 'text') else str(msg.content)
            print(f"[{step}]")
            if len(content) > 1000:
                print(f"  content: {content[:800]}...(중략)")
            else:
                print(f"  content: {content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                calls = []
                for tc in msg.tool_calls:
                    args_str = ", ".join(f"{k}={_format_arg(v)}" for k, v in tc['args'].items())
                    calls.append(f"{tc['name']}({args_str})")
                print(f"  tool_calls: {calls}")
            print()
            last_msg = msg
    return last_msg
```

---

## Part 7. 실전 시연 1 — 지역별 매출 분석 (단일 스킬)

```python
result = stream_agent(agent,
    {"messages": [{"role": "user", "content": "지역별 매출 현황을 분석해줘. 가장 매출이 높은 지역과 낮은 지역을 비교해줘."}]},
    config=skill_config,
)
```

**실행 트레이스 요약**:

| 순서 | 단계 | 내용 |
|---|---|---|
| 1 | `model` | `load_skill(skill_name=sales-analytics)` 호출 |
| 2 | `tools` | `customers`/`orders`/`order_items` 테이블 스키마 반환 |
| 3 | `model` | `run_command`로 지역별 매출 SQL 집계 실행 (`sales_analytics/query.py`) |
| 4 | `tools` | west/east/south/north 지역별 매출·주문수·AOV 반환 |
| 5 | `model` | 최고/최저 지역 비교 SQL 추가 실행 |
| 6 | `tools` | 최고(west)/최저(north) 매출 격차·배수 계산 결과 반환 |
| 7 | `model` | 매출 비중(%) 계산 SQL 추가 실행 → 최종 답변 생성 |

**최종 답변**:
```
도구로 orders 테이블에서 status='completed' 주문만 집계해, 지역(sales_region)별 매출 합계/주문수/AOV(평균주문금액)을 계산했습니다.

## 지역별 매출 현황(Completed 기준)

| 지역 | 완료 주문수 | 매출(총액) | 평균주문금액(AOV) | 매출 비중 |
|---|---:|---:|---:|---:|
| west | 27 | 57,311,400 | 2,122,644 | 31.84% |
| east | 25 | 50,193,035 | 2,007,721 | 27.88% |
| south | 26 | 41,015,410 | 1,577,516 | 22.78% |
| north | 18 | 31,494,440 | 1,749,691 | 17.50% |

## 가장 높은 지역 vs 가장 낮은 지역 비교
- 최고 매출 지역: west (57,311,400)
- 최저 매출 지역: north (31,494,440)

차이/배수:
- 매출 격차: 25,816,960
- 매출 배수: west가 north의 1.82배

구성요인(주문수 vs 객단가) 관점:
- 주문수: west 27건 vs north 18건 → west가 +9건
- AOV: west 2,122,644 vs north 1,749,691 → west가 약 +372,953(≈21.3%) 높음
→ 즉, west의 우위는 주문수가 더 많고(AOV도 더 높아서) 매출이 크게 벌어진 구조입니다.

원하시면 기간(최근 30/90일), 취소/환불 포함 여부, 또는 고객 티어/신규·기존 고객별로 지역 매출을 쪼개서도 비교해드릴게요.
```

> 💡 **에이전트가 자율적으로 수행한 것**: 사용자는 "지역별 매출 현황을 분석해줘"라고만 요청했지만, 에이전트는 (1) `sales-analytics` 스킬이 필요하다고 판단해 스스로 로드하고, (2) 스킬 문서에서 배운 테이블 구조(`orders.sales_region`, `orders.status`)를 근거로 SQL을 직접 작성하고, (3) 결과를 보고 "격차/배수/구성요인 분해"까지 추가 분석했습니다 — 이는 스킬이 "정답 SQL을 미리 정해둔 것"이 아니라 **"SQL을 올바르게 작성하기 위한 도메인 지식(스키마)"만 제공**하고, 실제 분석 논리는 LLM이 스스로 구성한다는 점을 보여줍니다.

---

## Part 8. 실전 시연 2 — 매출+재고 종합 보고서 (다중 스킬 조합)

```python
result2 = stream_agent(agent,
    {"messages": [{"role": "user", "content": "전체 매출과 재고 현황을 포함한 종합 보고서를 생성해줘. 보고서 가이드라인을 따라서 작성해."}]},
    config=skill_config,
)
```

**실행 트레이스 요약** (전체 약 20단계 중 핵심 단계):

| 순서 | 단계 | 내용 |
|---|---|---|
| 1 | `model` | `load_skill(customer-report)`, `load_skill(inventory-management)`, `load_skill(sales-analytics)`, `list_files(skills)` — **4개 도구를 한 번에 병렬 호출** |
| 2 | `tools` | 보고서 가이드라인 + 재고 DB 스키마 + `skills/` 폴더 목록(5개 스킬) + 매출 DB 스키마 반환 |
| 3 | `model` | `run_command(python skills/customer_report/generate.py --type full)` — 보고서 스킬의 **자체 생성 스크립트**를 먼저 실행 |
| 4 | `tools` | 보고서 초안(매출/재고 요약, "권장 액션"은 "(LLM이 작성)"으로 비워둔 템플릿) 반환 |
| 5~13 | `model`/`tools` (반복) | 월별 매출 추이, 고객 등급별 매출, 활성 고객 수, CLV 상위 고객, 이탈 위험 고객, 창고별 재고, 재주문 필요 상품, 재고 가치 상위 상품, 지역·등급별 매출 비중 등 **9회 이상의 SQL 쿼리**를 `sales_analytics/query.py` / `inventory_management/query.py`로 순차 실행 |
| 14 | `model` | `write_file(file_path=report_20260804_매출_재고_종합.md, content=...)` — 종합 결과를 마크다운 보고서로 저장 |
| 15 | `tools` | `파일 작성 완료: report_20260804_매출_재고_종합.md (3691자)` |
| 16 | `model` | 저장된 보고서 전문을 최종 답변으로 함께 출력 |

**최종 답변에 포함된 보고서 전문**:

```markdown
# 매출·재고 종합 보고서

> 생성일: 2026-08-04
> 데이터 범위(매출): 2024-01-02 ~ 2025-05-04 (completed 주문 기준)

## 1. 요약 (Executive Summary)
- 총 매출: 180,014,285원 (완료 주문 96건)
- 최고 매출 지역: west 57,311,400원(31.8%) / 최저 매출 지역: north 31,494,440원(17.5%)
- 고객(총 20명) 중 활성 고객: 16명(조건: active & 가입 90일 경과)
- 총 재고: 1,635개 / 총 재고 가치: 25,016,000원 (단종 제외)
- 재주문 필요 상품: 0개 (단종 제외, 전 창고 합산 재고가 reorder_point 이하인 상품 없음)

## 2. 매출 분석

### 2.1 기간별 매출 추이(월별)
[표 1] 2024-01(11건, 22,941,580원) ~ 2025-05(1건, 675,600원)까지 17개월 데이터

### 2.2 지역별 매출 분포
| 지역 | 완료 주문(건) | 매출(원) | 매출 비중 |
|---|---:|---:|---:|
| west | 27건 | 57,311,400원 | 31.8% |
| east | 25건 | 50,193,035원 | 27.9% |
| south | 26건 | 41,015,410원 | 22.8% |
| north | 18건 | 31,494,440원 | 17.5% |
- 최고(west) vs 최저(north) 매출 격차: 25,816,960원 (west가 1.82배)

### 2.3 고객 등급별 매출 분포
| 고객 등급 | 구매 고객(명) | 완료 주문(건) | 매출(원) | 매출 비중 |
|---|---:|---:|---:|---:|
| bronze | 11명 | 52건 | 91,608,455원 | 50.9% |
| gold | 3명 | 21건 | 44,707,970원 | 24.8% |
| silver | 6명 | 23건 | 43,697,860원 | 24.3% |

## 3. 고객 분석
- 전체 고객 20명 / 활성 고객 16명
- CLV(고객 생애가치) TOP 5: 전예린(gold, 20,902,030원) 등 5명
- 이탈 위험 고객(활성 & 90일 이상 미구매): 16명

## 4. 재고 현황
- 운영 상품 수(단종 제외) 14개, 총 재고 1,635개, 총 재고 가치 25,016,000원
- 창고별: 서울(623개, 용량사용률 12.5%), 부산(542개, 18.1%), 대전(563개, 28.2%)
- 재주문 필요 상품: 해당 없음(0개)
- 재고 가치 상위 TOP 5: 무선 이어폰(4,005,000원) 등

## 5. 권장 액션
1) north 지역 매출 개선 실험: 최저 매출 지역(17.5%)에 타겟 프로모션/채널 테스트, 주문수(18건) 확대를 1차 KPI로 설정
2) west 지역 성장 유지: 강점(객단가·주문수) 유지, 과도한 할인로 마진 훼손 방지
3) 브론즈 매출 의존도 완화: bronze 매출 비중 50.9% → silver/gold 업셀 유도
4) 이탈 위험 고객 재활성화: 최근 구매일 기준 세그먼트별 리마인드·재구매 쿠폰 차등 운영
5) 재고자산 집중 관리(가치 상위 5개): 별도 수요예측·안전재고 정책으로 품절/과잉 동시 방지
```

> 💡 **다중 스킬 조합의 실제 동작**: 하나의 자연어 요청("종합 보고서를 생성해줘")에 대해 에이전트는 **`customer-report`(보고서 형식) + `sales-analytics`(매출 SQL) + `inventory-management`(재고 SQL)** 3개 스킬을 순서대로 필요할 때마다 로드했습니다. 흥미롭게도 `customer_report/generate.py`(스킬 자체의 자동 생성 스크립트)를 먼저 실행해 **초안**을 얻은 뒤, 그 초안에서 비어 있던 부분(권장 액션, 세부 월별 추이, CLV 등)을 **직접 SQL을 추가로 실행해 채워 넣는 하이브리드 방식**을 택했습니다 — 스킬이 제공하는 스크립트를 "그대로 믿고 끝내는" 것이 아니라, 스크립트 결과를 검토하고 부족한 부분을 스스로 보완하는 능동적인 문제 해결 과정을 보여줍니다.

---

## Part 9. [실습] 새로운 Skill 만들기 — `clock` 스킬

> 실습에서 다루지 않은 스킬들도 프롬프트를 통해 실행해 보세요. 새로운 스킬을 추가하고, Agent로 테스트해보세요.
> 1. `skills/` 아래에 새 폴더 생성
> 2. `SKILL.md` 작성 (YAML frontmatter 형식)
> 3. 부가적인 파일 작성
>
> Skill Creator Skill를 사용해서 새로운 스킬을 만들 수 있습니다.

### 실습: `skill-creator` 스킬을 이용해 `clock` 스킬 생성 요청

```python
agent = create_agent(
    llm, tools=tools,
    system_prompt=f"""도구를 활용할 때마다, 중간 과정을 간략히 설명하세요.
{format_skill_list(scan_skills())}""",
    checkpointer=InMemorySaver(),
)

skill_config = {'configurable': {'thread_id': 'skill-agent'}}
```

```python
result = stream_agent(agent,
    {"messages": [{"role": "user", "content": """
    지역별 현재 시간을 초단위로 알려주는 파이썬 스크립트 기반의 'clock'스킬을 만들어 줘.
    스킬 설명은 최대한 간략하게 하고. 기본값은 서울.
    스킬 잘 만들어졌는지 테스트도 해줘.
    """}]},
    config=skill_config,
)
```

**실행 트레이스 (요약)**:
1. `list_files('.')` , `list_files('skills')`로 현황 파악
2. `load_skill('skill-creator')` — "스킬을 만들고, 초안 작성 → 테스트 프롬프트 실행 → 평가"라는 스킬 제작 절차 자체를 로드
3. `write_file('skills/clock/SKILL.md', ...)` — YAML frontmatter(`name: clock`, `description: ...`) + 사용 방식/스크립트 사용법 작성
4. `write_file('skills/clock/scripts/clock.py', ...)` — `zoneinfo` 기반 다지역 시간 조회 스크립트 작성
5. `run_command('python --version && python skills/clock/scripts/clock.py | head')` → **서울 기본값 정상 동작 확인**: `Seoul  Asia/Seoul  2026-08-04 17:55:00`

### ⚠️ 실제로 발견된 이슈 — 안전 필터의 오탐(False Positive)

```python
run_command(command='python skills/clock/scripts/clock.py Seoul Tokyo London NYC UTC --format "%Y-%m-%d %H:%M:%S %Z%z"')
```

**실행 결과**:
```
[차단] 위험한 명령 패턴(\b(format|mkfs|fdisk)\b)이 포함되어 있습니다.
```

> 🐛 **오탐 원인**: Part 2에서 만든 `BLOCKED_PATTERNS`의 `r'\b(format|mkfs|fdisk)\b'`는 디스크를 포맷하는 `format`/`mkfs`/`fdisk` **명령어**를 막기 위한 패턴입니다. 그런데 에이전트가 스크립트에 추가한 `--format` **CLI 옵션 이름**에도 단어 경계(`\b`) 매칭으로 걸려버려, **전혀 위험하지 않은 정상적인 명령까지 차단**되었습니다. 이는 노트북 Part 2의 이론("문자열 패턴 검사는 표현을 바꾸면 통과할 수 있어, 완전한 격리가 되지는 않습니다")이 경고한 것과 **정반대 방향의 실패**입니다 — 공격을 놓치는 것(false negative)이 아니라, **안전한 명령을 막는 것(false positive)**도 패턴 기반 필터링의 실제 문제임을 보여줍니다.
>
> **에이전트의 실제 대응**: 차단 메시지를 받은 에이전트는 스스로 옵션 이름을 `--format`에서 **`--fmt`로 변경**하여 스크립트와 명령을 다시 작성했고, 이후 정상적으로 실행되었습니다.

```python
run_command(command='python skills/clock/scripts/clock.py Seoul Tokyo London NYC UTC --fmt "%Y-%m-%d %H:%M:%S %Z%z"')
```

**실행 결과**:
```
Seoul	Asia/Seoul	2026-08-04 17:55:18 KST+0900
Tokyo	Asia/Tokyo	2026-08-04 17:55:18 JST+0900
London	Europe/London	2026-08-04 09:55:18 BST+0100
NYC	America/New_York	2026-08-04 04:55:18 EDT-0400
UTC	UTC	2026-08-04 08:55:18 UTC+0000
```

### 실습: 예외 상황 테스트와 평가 데이터셋 작성

```python
run_command(command="python skills/clock/scripts/clock.py Europe/Paris Invalid/Zone 2> /tmp/clock_err.txt; echo 'STDERR:'; cat /tmp/clock_err.txt")
```

**실행 결과**:
```
Europe/Paris	Europe/Paris	2026-08-04 10:55:20
STDERR:
Invalid/Zone	ERROR	'No time zone found with key Invalid/Zone'
```

> 💡 존재하지 않는 타임존(`Invalid/Zone`)을 입력해도 스크립트가 **죽지 않고 오류 메시지를 STDERR로만 남기고, 나머지 유효한 지역은 정상 출력**했습니다 — 에이전트가 의도적으로 이런 예외 케이스까지 테스트한 것으로 보입니다.

```python
run_command(command="""python -c "import json;print(json.load(open('skills/clock/evals/evals.json'))['skill_name'])")""")
```

이 셀 이전에 `write_file('skills/clock/evals/evals.json', ...)`로 **3개의 테스트 프롬프트가 담긴 평가 데이터셋**을 함께 생성했습니다 — README에는 없는 내용이지만, `load_skill('skill-creator')`가 로드한 "스킬 제작 절차"(초안 → 테스트 → 평가)를 그대로 따른 것입니다. `evals.json`이 정상적으로 저장되었는지 `skill_name` 필드를 읽어 확인한 결과 `clock`이 출력되었습니다.

**최종 답변**:
```
아래처럼 파이썬 스크립트 기반 clock 스킬을 새로 만들었어. (기본값: 서울/Asia/Seoul, 초 단위 출력)

## 1) 생성된 스킬 파일
- skills/clock/SKILL.md : 스킬 설명(최대한 간략)
- skills/clock/scripts/clock.py : 실행 스크립트
- skills/clock/evals/evals.json : 간단 테스트 프롬프트 3개

## 2) 스크립트 사용법
python skills/clock/scripts/clock.py                     # 기본(서울)
python skills/clock/scripts/clock.py Seoul Tokyo London NYC UTC   # 여러 지역
python skills/clock/scripts/clock.py --fmt "%Y-%m-%d %H:%M:%S %Z%z" Seoul  # 포맷 지정

## 3) 테스트(직접 실행으로 검증)
도구(터미널)로 실제 실행해서 동작 확인했어. 기본값/다중 지역/에러 처리 케이스를 확인.
```

### 실습: 새 스킬을 완전히 새로운 대화(스레드)에서 자동 로드해보기

```python
agent = create_agent(
    llm, tools=tools,
    system_prompt=f"""도구를 활용할 때마다, 중간 과정을 간략히 설명하세요.
{format_skill_list(scan_skills())}""",
    checkpointer=InMemorySaver(),
)

skill_config = {'configurable': {'thread_id': 'skill-agent'}}
```

```python
result = stream_agent(agent,
    {"messages": [{"role": "user", "content": """
    오늘 오후 6시까지 몇초 남았니?
    """}]},
    config=skill_config,
)
```

**실행 트레이스**:
1. `load_skill(skill_name=clock)` → 방금 만든 `clock` 스킬의 사용법(별칭 규칙, 스크립트 경로)을 로드
2. `run_command`로 `zoneinfo` 기반 파이썬 코드를 즉석에서 작성하여 "현재 시각"과 "오늘 18:00:00"의 차이를 직접 계산 (저장된 `clock.py` 스크립트를 그대로 쓰지 않고, 스킬 지식을 참고해 **새로운 계산 코드를 즉석에서 작성**)
3. 결과: `2026-08-04T17:57:15.812418+09:00`, `2026-08-04T18:00:00+09:00`, `164.187582`

**최종 답변**: `(서울 기준) 현재 시각은 2026-08-04 17:57:15이고, 오늘 18:00:00까지 남은 시간은 약 164.19초(= 2분 44.19초)입니다.`

> 💡 **가장 중요한 확인 사항**: `agent`를 **완전히 새로 생성**했음에도(이전 대화 맥락 없음), `scan_skills()`가 매번 `skills/` 폴더를 다시 읽으므로 **방금 만든 `clock` 스킬이 시스템 프롬프트의 스킬 목록에 즉시 포함**되었고, 에이전트는 "지금 몇 초 남았니?"라는 질문에서 `clock` 스킬을 스스로 연결지어 로드했습니다 — README p.74의 "이후, 해당 스킬을 로드(read)하면, 뒷부분의 내용을 컨텍스트로 가져와 활용하는 방식"이 **새로 만든 스킬에도 즉시 적용**됨을 실제로 증명한 사례입니다. 별도의 재배포나 에이전트 재구성 없이 "스킬 폴더에 파일을 추가하는 것"만으로 에이전트의 능력이 확장되었습니다.

---

## Part 10. 정리

### 10.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | `run_command` 안전장치 | 비밀파일 차단·위험명령 차단·cwd 고정 3중 안전장치 구현 |
| 2 | 셸 도구만으로 시각화 | 컬럼 탐색 → 집계+시각화 스크립트 작성 → `result.png` 저장까지 자율 수행 |
| 3 | `scan_skills()` | 5개 스킬의 프론트매터만 파싱해 짧은 목록 생성 (Context Bloat 방지) |
| 4 | `load_skill` | 필요한 스킬의 전체 본문(스키마, 규칙, 스크립트 경로)을 그때 로드 |
| 5 | 단일 스킬 활용 | `sales-analytics` 로드 → SQL 3회 실행 → 지역별 매출 비교 분석 |
| 6 | 다중 스킬 조합 | `customer-report`+`sales-analytics`+`inventory-management` 3종 조합, SQL 9회+, `write_file`로 보고서 저장 |
| 7 | **안전 필터 오탐** | `--format` 옵션이 `format` 위험 패턴에 걸려 차단 → 에이전트가 `--fmt`로 우회(실제 발생) |
| 8 | [실습] 새 스킬 생성 | `skill-creator` 스킬 활용 → SKILL.md/스크립트/evals.json 작성 → 직접 테스트 |
| 9 | 새 스킬 즉시 인식 | 완전히 새로운 대화에서도 방금 만든 `clock` 스킬을 자동으로 로드·활용 |

### 10.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 확인한 것 |
|---|---|
| Claude Code Skills — Context Bloat(p.72) | `scan_skills()`가 프론트매터만 시스템 프롬프트에 주입해 컨텍스트 절약 |
| Skill 구성요소(p.73) | `SKILL.md` + `scripts/`(cli-tool) + `evals/` 폴더 구조로 5개+1개(clock) 스킬 실습 |
| SKILL.md — YAML Frontmatter + 점진적 공개(p.74) | `name`/`description` 프론트매터 우선 노출, `load_skill()`로 본문 지연 로드 |
| Agent — Planning/Tool Executing/Observation(p.69) | 셸 도구 하나로 탐색→코드작성→실행→검증의 전체 루프를 자율 수행 |
| LLM Tool Calling — Taker에서 Giver로(p.66) | `write_file`로 종합 보고서를 실제 파일로 저장(README 예시와 동일한 패턴) |
| SQL in LangChain — 안전 문제(p.61) | 셸 명령 실행에도 동일한 원리(비밀파일/위험명령 차단)가 확장 적용됨을 확인 |

### 10.3 참고 자료

- Anthropic Agent Skills 공식 문서: <https://www.anthropic.com/news/skills>
- Python `zoneinfo` (표준 라이브러리 타임존): <https://docs.python.org/3/library/zoneinfo.html>
- `koreanize_matplotlib`: <https://github.com/ychoi-kr/koreanize-matplotlib>
- 원본 실습 노트북: `[실습]_9_Code_Interpreter를_이용한_데이터_시각화와_Skill.ipynb`

### 10.4 다음 단계

- **Lab 8** (`[실습]_8_LangChain과_다양한_툴_연동.ipynb`)의 "여러 개별 Tool 연결" 방식과, 이번 Lab 9의 "셸 도구 하나 + Skill" 방식을 비교하면, 도구 수가 늘어날 때 **컨텍스트 관리 전략을 어떻게 바꿔야 하는지**에 대한 실전적 감을 얻을 수 있습니다.
- 이번 실습에서 발견한 **안전 필터 오탐(false positive)** 문제는, 문자열 패턴 기반 보안 규칙을 설계할 때 "위험한 것을 놓치지 않는 것"과 "정상적인 것을 막지 않는 것" 사이의 트레이드오프를 항상 함께 고려해야 함을 보여줍니다 — 정규식에 `\b` 단어 경계를 썼음에도 옵션 이름(`--format`)까지 걸린 것처럼, 패턴은 항상 실제 사용 사례로 검증해야 합니다.
- 실무 팁: Skill을 프로덕션에 도입한다면 (1) `SKILL.md`의 `description`을 정확하고 구체적으로 작성해 에이전트의 자동 로드 판단 정확도를 높이고, (2) 이번 실습의 `evals.json`처럼 **스킬별 최소한의 회귀 테스트**를 함께 관리하며, (3) 스킬 스크립트에도 이번 실습의 `run_command`/`safe_path` 같은 **셸/파일 안전장치를 동일하게 적용**해야 합니다 — 스킬은 결국 "LLM이 자유롭게 실행하는 코드"이므로, Tool 하나에 적용하는 보안 원칙이 스킬 전체에도 그대로 적용됩니다.
