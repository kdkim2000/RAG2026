# Lab 8. LangChain과 다양한 툴 연동하기

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"LangChain 실용 예제"** 챕터(p.59~71 이론)와 `[실습]_8_LangChain과_다양한_툴_연동.ipynb`의 **실제 실행 결과**를 결합하여 재구성한 것입니다. 모든 코드와 실행 결과는 Windows 로컬 환경(venv, Python 3.12, `gpt-5.2` / 이미지 생성 `gpt-image-2`)에서 실제로 실행하여 얻은 것이며, 누락 없이 포함되어 있습니다.

## 학습 목표

- **Tool(≈ Function)**의 개념을 이해하고, LangChain에서 Tool을 만들고 LLM에 연결한다.
- LLM이 언제 Tool을 호출하는지(`tool_calls`)를 실제로 관찰하고, Tool 실행 결과를 다시 LLM에 전달하는 흐름을 구현한다.
- `create_agent()`로 **Agent**(Tool Calling을 반복하는 구조)를 만들고, 스트리밍으로 내부 동작을 관찰한다.
- **Checkpointer**로 대화 메모리를 연결하고, `thread_id`로 대화 세션을 구분한다.
- 파일 시스템 접근(`read_file`/`write_file`), 웹 페이지 조회(`fetch_url`), 이미지 생성(`draw_image`) 등 실전적인 Tool을 직접 만든다.
- 여러 Tool을 조합해야 하는 복합적인 요청(실시간 인포그래픽 생성, 멀티턴 대화로 코드 패키지 구축)을 Agent가 처리하는 과정을 확인하고, 실전에서 만날 수 있는 이슈(타임아웃 등)에 대응한다.

---

## Part 1. Tool과 Function Calling

### 1.1 이론: Tool이란 무엇인가 (교재 p.63)

**Tool (≈ Function)**: LLM이 외부 함수를 이용할 수 있도록 구현된 모듈입니다. LLM이 함수 실행 포맷의 출력을 생성할 수 있는 능력을 사전에 학습해야 하며(Function Calling 튜닝), LangChain은 이를 표준화된 인터페이스로 감싸줍니다.

**LangChain의 주요 Tool**
- **SerpAPI**: 구글, 네이버, 유튜브 등의 검색 API를 지원하는 서비스
- **Tavily**: AI 기반 검색 API — 본 실습에서 사용
- **PythonREPLTool()**: 파이썬 코드를 실행하는 기능 (ChatGPT Code Interpreter와 다르게, 로컬에서 실행)

> 💡 **참고**: LangChain의 Tool은 OpenAI의 Tool(Function Calling)과 유사한 개념입니다. LLM은 Tool에 입력을 전달하고, 답변을 파싱하는 작업을 수행합니다.

### 1.2 실습: 환경 설정

패키지 설치와 LLM 초기화입니다.

```python
!pip install openai matplotlib langchain langchain-openai langchain-community langchain-tavily langgraph -q
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(model = 'gpt-5.2', reasoning_effort='low')
```

> **실행 결과**: 두 셀 모두 정상 실행(출력 없음). `.env`에 `OPENAI_API_KEY`, `TAVILY_API_KEY`가 설정되어 있어야 합니다.

### 1.3 실습: Tavily Search 툴 사용하기

Tavily Search([tavily.com](https://tavily.com/))는 한 달에 1,000회 검색을 지원하는 검색 API로, 구글 계정으로 회원가입할 수 있습니다.

```python
from langchain_tavily import TavilySearch

# TAVILY : 웹 검색을 수행하는 툴
# TAVILY_API_KEY https://app.tavily.com/home
```

```python
load_dotenv(override=True)

tavily_search = TavilySearch(max_results=3)

tavily_search.invoke("Gemma 4")
```

**실행 결과**:
```python
{'query': 'Gemma 4', 'follow_up_questions': None, 'answer': None, 'images': [],
 'results': [
   {'url': 'https://lmstudio.ai/models/gemma-4', 'title': 'Gemma 4',
    'content': "Gemma is a family of open models built by Google DeepMind. Gemma 4 models are "
                "multimodal, handling text and image input ... context window of up to 256K tokens "
                "and maintains multilingual support in over 140 languages. ... Available in four "
                "sizes: E2B, E4B, 26B A4B, and 31B.",
    'score': 0.9245858, 'raw_content': None, 'id': 'cce4d9-00'},
   {'url': 'https://arxiv.org/html/2607.02770v1', 'title': 'Gemma 4 Technical Report', ...},
   {'url': 'https://huggingface.co/blog/gemma4', 'title': 'Welcome Gemma 4: ...', ...}
 ],
 'response_time': 0.72, 'request_id': '5b15f780-d97b-4ba9-a980-a382084535f0'}
```

실제 웹 검색이 정상적으로 수행되었으며, 3개의 관련 결과가 점수(score)와 함께 반환되었습니다.

이어서, 검색 결과를 사람이 읽기 좋은 형태로 출력하는 예제입니다.

```python
print('Tavily Search 예제:')
example_query = "Tavily Search 사용법"
search_results = tavily_search.invoke(example_query)
print(f"검색어: '{example_query}'")
for i, result in enumerate(search_results['results']):
    print(f"  결과 {i+1}:")
    print(f"    URL: {result['url']}")
    print(f"    제목: {result['title']}")
    print(f"    내용: {result['content'][:200]}...") # 내용이 길면 일부만 출력
```

**실행 결과**:
```
Tavily Search 예제:
검색어: 'Tavily Search 사용법'
  결과 1:
    URL: https://docs.tavily.com/documentation/best-practices/best-practices-search
    제목: Best Practices for Search - Tavily Docs
    내용: Run queries in parallel, but cap how many run at once with a semaphore so you stay under
          your rate limit. Handle failures per query. Tag each result `ok` / `error` and retry
          only the failed ones wi...
  결과 2:
    URL: https://velog.io/@woody_ahn/...
    제목: Tavily: LLM을 위한 최적화된 검색 API
    내용: # Tavily 소개
          Tavily Search API는 LLM과 RAG 시스템에 최적화된 검색 엔진입니다 ...
  결과 3:
    URL: https://www.freecodecamp.org/news/how-to-add-real-time-web-search-to-your-llm-using-tavily
    제목: How to Add Real-Time Web Search to Your LLM Using Tavily
    내용: 1. The user sends a question. The agent receives the message and determines that it
          needs external information. 2. Tavily performs a search ...
```

### 1.4 실습: Custom Tool 만들기 (`calculator`, `current_date`)

`@tool` 데코레이터로 임의의 파이썬 함수를 LLM이 호출 가능한 Tool로 변환할 수 있습니다. **docstring이 곧 Tool의 설명(description)이 되어 LLM에게 전달**되므로, 명확하게 작성해야 합니다.

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """수학 계산식을 입력받아 계산 결과를 반환합니다.
    사칙연산, 거듭제곱(**) 등 Python 수식을 지원합니다.

    Args:
        expression: 계산할 수식 (예: '3 + 5 * 2', '2 ** 10')
    """
    try:
        allowed_chars = set('0123456789+-*/.()**% ')
        if not all(c in allowed_chars for c in expression):
            return f'허용되지 않는 문자가 포함되어 있습니다: {expression}'
        # 주의: 교육용 코드입니다. 프로덕션에서는 ast.literal_eval 또는 numexpr을 사용하세요.
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f'계산 오류: {str(e)}'

print(f"calculator('3 + 5 * 2') = {calculator.invoke({'expression': '3 + 5 * 2'})}")

@tool
def current_date() -> str:
    "현재 날짜를 %y-%m-%d 형식으로 반환합니다."
    from datetime import datetime
    return f'현재 날짜는 {datetime.now().strftime("%Y-%m-%d")} 입니다!'

print(current_date.invoke({}))
```

**실행 결과**:
```
calculator('3 + 5 * 2') = 13
현재 날짜는 2026-08-07 입니다!
```

> ⚠️ **보안 주의**: `calculator`는 `eval()`을 사용하는 **교육용 코드**입니다. 허용 문자 화이트리스트로 최소한의 방어를 하고 있지만, 실제 서비스에서는 `ast.literal_eval` 또는 `numexpr` 같은 안전한 수식 평가 라이브러리를 사용해야 합니다.

세 개의 Tool을 리스트로 묶어 확인합니다.

```python
from rich import print as rprint

tools = [tavily_search, calculator, current_date]

rprint(tools)
```

**실행 결과**:
```
[
    TavilySearch(max_results=3, api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr('**********'), api_base_url=None)),
    StructuredTool(
        name='calculator',
        description="수학 계산식을 입력받아 계산 결과를 반환합니다.\n    사칙연산, 거듭제곱(**) 등 Python 수식을 지원합니다.\n\n    Args:\n        expression: 계산할 수식 (예: '3 + 5 * 2', '2 ** 10')",
        args_schema=<class 'langchain_core.utils.pydantic.calculator'>,
        func=<function calculator at 0x00000276424F1440>
    ),
    StructuredTool(
        name='current_date',
        description='현재 날짜를 %y-%m-%d 형식으로 반환합니다.',
        args_schema=<class 'langchain_core.utils.pydantic.current_date'>,
        func=<function current_date at 0x0000027646639D00>
    )
]
```

API 키(`tavily_api_key`)는 `SecretStr`로 마스킹되어 출력됨을 확인할 수 있습니다 — 로그에 실수로 키가 노출되는 것을 방지하는 LangChain의 기본 동작입니다.

### 1.5 이론: Tool Binding과 Tool Calling 흐름 (교재 p.64~66)

LLM이 Tool을 직접 "제어"할 수는 없지만, **제어 명령을 텍스트로 생성**할 수 있습니다. 이 개념을 자동화 파이프라인에 연결한 것이 Anthropic의 Computer Use Agent, OpenAI Agent Mode 같은 시스템입니다.

> **예시**: User: "61번 서버 상태 알려줄래?" → LLM: `Call {Tool: Retrieve_Status, Query: '61'}`

Tool이 탑재된 LLM의 판단 흐름:
1. Tool이 필요한 경우, Tool 실행을 요청 (`tool_calls` 생성)
2. 프로그램상에서 Tool 요청을 파싱하여 실제 실행
3. Tool 실행 결과(**Tool Message**)를 LLM에게 다시 전달
4. LLM은 이를 바탕으로 최종 답변 생성

> **예시**: User: "61번 서버 상태 알려줄래?" + LLM: `Call {Tool: Retrieve_Status, Query: '61'}` + Tool: `"504"` → "504 Error로, 게이트웨이 타임아웃으로 추정됩니다."

이 패턴이 일반화되면, **외부 Tool과 Text로 소통하는 인터페이스만 있으면 LLM이 양방향으로 세상과 상호작용**할 수 있습니다 — 파일 시스템 접근, API 호출, Github 이슈 해결 및 커밋 등. 이를 "**Taker에서 Giver로**"라고 표현합니다.

> **예시**: User: "(중략) 정리해서 report.txt에 저장해줘." → LLM: `Function Call {Tool: Write_file, Content: '# 최종 정리 내용 …'}` → Tool: `{"Result":"저장 완료!"}` → "리포트 저장이 완료되었습니다!"

(이 시나리오는 Part 4에서 `write_file` Tool로 직접 구현합니다.)

### 1.6 실습: LLM에 Tool Binding하고 `tool_calls` 확인하기

`bind_tools()`로 LLM에 Tool 스키마를 연결합니다.

```python
llm_with_tools = llm.bind_tools(tools)
llm_with_tools
rprint(llm_with_tools)
```

**실행 결과** (가독성을 위해 rich의 상세 컬러 덤프 중 핵심만 축약):
```python
_ChatModelBinding(
    bound=ChatOpenAI(
        model_name='gpt-5.2',
        reasoning_effort='low',
        profile={'name': 'GPT-5.2', 'max_input_tokens': 272000, 'max_output_tokens': 128000,
                 'tool_calling': True, 'reasoning_effort_levels': ['none','low','medium','high','xhigh']}
    ),
    kwargs={
        'tools': [
            {'type': 'function', 'function': {'name': 'tavily_search', 'description': 'A search engine optimized for...', ...}},
            {'type': 'function', 'function': {'name': 'calculator', 'description': "수학 계산식을 입력받아...", ...}},
            {'type': 'function', 'function': {'name': 'current_date', 'description': '현재 날짜를 %y-%m-%d 형식으로 반환합니다.', ...}}
        ]
    }
)
```

이제 다양한 질문으로 **LLM이 언제 Tool을 호출하는지** 직접 관찰합니다. `llm_with_tools`는 여전히 `llm`과 같은 `Runnable`이므로, `invoke()`로 호출합니다.

| 질문 | 결과 | 해석 |
|---|---|---|
| `"안녕?"` | `content='안녕하세요! 무엇을 도와드릴까요?'`, `tool_calls=[]` | 일반 인사 → Tool 불필요 |
| `"너는 무슨 도구가 잇는지 소개해 줘"` | 3개 도구를 텍스트로 설명, `tool_calls=[]` | 도구 목록은 시스템 스키마에 이미 있으므로 Tool 실행 불필요 |
| `"29392 * 23919는 뭐야?"` | `content=''`, `tool_calls=[{'name': 'calculator', 'args': {'expression': '29392 * 23919'}, ...}]` | 정확한 계산이 필요 → **calculator 호출** |
| `"양자 컴퓨터의 정의가 뭐야?"` | 일반 지식으로 직접 답변, `tool_calls=[]` | 사전 학습 지식으로 충분 → Tool 불필요 |
| `"양자 컴퓨터 최신 소식 있어?"` | `content=''`, `tool_calls=[{'name': 'tavily_search', 'args': {'query': '2026 양자 컴퓨터 최신 소식 ...', 'search_depth': 'advanced', 'time_range': 'month'}, ...}]` | 최신 정보 필요 → **tavily_search 호출** |

실제 코드와 결과:

```python
# 툴 실행을 하지 않음
llm_with_tools.invoke("안녕?")
# AIMessage(content='안녕하세요! 무엇을 도와드릴까요?', tool_calls=[], ...)

llm_with_tools.invoke("너는 무슨 도구가 잇는지 소개해 줘")
# AIMessage(content='내가 여기서 쓸 수 있는 "도구(tools)"는 아래 3가지야. ...', tool_calls=[], ...)

llm_with_tools.invoke("29392 * 23919는 뭐야?")
# AIMessage(content='', tool_calls=[{'name': 'calculator', 'args': {'expression': '29392 * 23919'},
#            'id': 'call_4GOK59EagcuDfaOVJlhHcC0t', 'type': 'tool_call'}], ...)

# 툴 실행을 하지 않을 확률이 높음
llm_with_tools.invoke("양자 컴퓨터의 정의가 뭐야?")
# AIMessage(content='양자 컴퓨터(quantum computer)는 정보를 "큐비트(qubit)"로 표현하고 ...', tool_calls=[], ...)

llm_with_tools.invoke("양자 컴퓨터 최신 소식 있어?")
# AIMessage(content='', tool_calls=[{'name': 'tavily_search',
#            'args': {'query': '2026 양자 컴퓨터 최신 소식 오류 정정 논리 큐비트 IBM Google IonQ Quantinuum 2025 2026',
#                      'search_depth': 'advanced', 'topic': 'general', 'time_range': 'month'},
#            'id': 'call_arl7LbhEOgh5gdHt4SP0yVUI', 'type': 'tool_call'}], ...)
```

`content`를 생성하는 대신 `tool_calls`가 도출된 것을 확인할 수 있습니다.

```python
llm_with_tools.invoke("29392 * 23919는 뭐야?").tool_calls
# [{'name': 'calculator', 'args': {'expression': '29392*23919'}, 'id': 'call_ufKGoipdJvSEmIzK1rXCI0Be', 'type': 'tool_call'}]

llm_with_tools.invoke("오늘 운동을 빠질까요?").tool_calls
# No Tool Call
# []
```

### 1.7 실습: Tool 실행 결과를 다시 LLM에 전달하기

`tool_call`(dict)을 그대로 `Tool.invoke()`에 넣으면, 결과가 **`ToolMessage`**로 반환됩니다 (`tool_call_id`가 함께 보존되어 어떤 호출에 대한 응답인지 추적 가능).

```python
calculator.invoke(llm_with_tools.invoke("29392 * 23919는 뭐야?").tool_calls[0])
```

**실행 결과**:
```python
ToolMessage(content='703027248', name='calculator', tool_call_id='call_9WB72gb2ySauG7kXgtm3kFnP')
```

이제 `[HumanMessage, AIMessage(tool_calls 포함), ToolMessage]` 3개를 순서대로 LLM에 다시 전달하면, LLM이 Tool 결과를 반영한 **최종 답변**을 생성합니다.

```python
# Message 전달하기 예시
from langchain.messages import HumanMessage, AIMessage, ToolMessage

question = '29392 * 23919가 뭐야?'
tool_call_msg = llm_with_tools.invoke(question)
tool_msg = calculator.invoke(tool_call_msg.tool_calls[0])

msgs = [HumanMessage(question),
       tool_call_msg,
       tool_msg]

llm_with_tools.invoke(msgs)
```

**실행 결과**:
```python
AIMessage(content='29392 × 23919 = **703,027,248** 입니다.', tool_calls=[], ...)
```

계산 결과(703,027,248)가 정확히 반영된 최종 답변이 생성되었습니다. **이 4단계(LLM 호출 → tool_calls 확인 → Tool 실행 → 결과를 메시지에 추가해 재호출)를 자동으로 반복해주는 것이 바로 다음 Part의 "Agent"입니다.**

### 1.8 이론: LangChain에서 커스텀 Tool 만들기 (교재 p.67)

Custom 함수의 경우에도, decorator를 통해 tool로 변환 가능합니다. Argument 정보 및 Description이 필요합니다.

```python
from langchain_core.tools import tool

@tool
def multiply(x: int, y: int) -> int:
    "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
    return x * y

example = {'x': 10, 'y': 99}
multiply.invoke(example)
```

> 이 예제는 교재 원문의 것으로, 위에서 만든 `calculator`/`current_date`와 동일한 `@tool` 데코레이터 패턴을 보여줍니다.

---

## Part 2. Agent — Tool을 스스로 활용하는 LLM

### 2.1 이론: Agent란? (교재 p.69~70)

**Agent**는 위에서 수동으로 반복했던 Tool Calling 과정(`llm 실행 → Tool 요청 → Tool 실행 결과 전달 → llm 실행 → ...`)을 **자동으로 반복하여 작업을 완료하는 구조**입니다. Tool의 결과물을 얻는 것에서 끝나지 않고, 해당 Tool을 사용해 문제를 해결합니다.

Agent는 **Planning(계획), Tool Executing(도구 실행), Observation(관찰)**을 포함하며, 구성 방식에 따라 다양한 범위의 작업을 수행할 수 있습니다.

**ReAct 패턴** (*ReAct: Synergizing Reasoning and Acting in Language Models*, [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629))은 "생각 → 행동 → 관찰"을 반복하며 다음 행동을 계획하고 실행하는 대표적인 Agent 프롬프트 방식입니다.

```
프롬프트: '강아지와 고양이 중 누가 더 평균적으로 똑똑한가요? 뇌세포의 수를 비교하세요.'

응답:
생각: 두 동물의 뇌세포 수를 비교하는 것이 중요하다.
행동: {'검색', '검색어: 개와 고양이의 뇌세포 비교'}
관찰: URL) https://dogs-cats-brains-neurons-study-vanderbilt/ (…) cats have 250 million
      cortical neurons, while dogs have about 530 million. (…)
생각: 250 million보다 530 million이 더 많은 뇌세포 수를 가진 것으로 보아 강아지가
      고양이보다 더 똑똑한 것 같다.
최종 답변: 강아지가 평균적으로 더 많은 뇌세포 수를 가지고 있으므로 고양이보다 똑똑합니다.
```

`langchain 1.0`의 `create_agent()`는 이 ReAct류 루프를 몇 줄로 구현해줍니다.

### 2.2 실습: `create_agent`로 기본 에이전트 만들기

```python
tools = [calculator, tavily_search, current_date]
```

```python
from langchain.agents import create_agent

system_prompt='''툴을 사용하기 전, 툴 사용 계획에 대해 설명하고 중간 단계를 매번 간략하게 요약하세요..'''

agent = create_agent(llm, tools=tools, system_prompt= system_prompt)

agent
```

**실행 결과**:
```
<langgraph.graph.state.CompiledStateGraph object at 0x000002764687A690>
```

```python
type(agent)
# <class 'langgraph.graph.state.CompiledStateGraph'>
```

`create_agent()`는 **LangGraph의 `CompiledStateGraph`**를 반환합니다 — Agent가 내부적으로 그래프(상태 기계) 구조로 구현되어 있음을 보여줍니다.

### 2.3 실습: 에이전트 실행해보기

**(1) 단순 Tool 호출: 오늘 날짜 확인**

```python
question = "오늘 날짜를 확인해줘."

response = agent.invoke({'messages':[HumanMessage(question)]})

response['messages'][-1].text
```

**실행 결과**: `'오늘 날짜는 **2026-08-07** 입니다.'`

전체 메시지 흐름을 보면, 시스템 프롬프트의 지시("툴 사용 계획을 설명하라")를 따라 계획을 먼저 설명한 뒤 Tool을 호출한 것을 알 수 있습니다.

```python
response
```

```python
{'messages': [
  HumanMessage(content='오늘 날짜를 확인해줘.'),
  AIMessage(content='툴 사용 계획: 시스템에서 제공하는 `current_date` 툴을 호출해 오늘 날짜를 확인한 뒤, 결과를 그대로 알려드리겠습니다.',
            tool_calls=[{'name': 'current_date', 'args': {}, 'id': 'call_ZTBUcG8oWAu24SMzliKeAubm', ...}]),
  ToolMessage(content='현재 날짜는 2026-08-07 입니다!', name='current_date', tool_call_id='call_ZTBUcG8oWAu24SMzliKeAubm'),
  AIMessage(content='오늘 날짜는 **2026-08-07** 입니다.', tool_calls=[])
]}
```

**(2) 복합 Tool 호출: 실시간 리서치**

```python
question = "2026년 7월 출시된 DeepSeek V4 Flash 모델의 장단점을 조사해서 마크다운 표로 작성해줘."

response = agent.invoke({'messages':[HumanMessage(question)]})

response['messages'][-1].text
```

**실행 결과** (요약):

> 중간 단계 요약: (1) 2026-07-31 DeepSeek API Change Log와 Hugging Face 모델 카드를 우선 확인했고, (2) 가격/컨텍스트/운영 특성과 제3자 코멘트를 교차로 모아 장단점을 묶었습니다.
>
> ### DeepSeek V4 Flash 장단점
>
> | 구분 | 내용(요약) | 근거/출처 |
> |---|---|---|
> | 장점 | 에이전트/도구사용 벤치마크 성능이 크게 향상 | DeepSeek API Change Log(2026-07-31) |
> | 장점 | 가성비(토큰 단가)가 매우 낮음: 입력(캐시 미스) $0.14/1M, 출력 $0.28/1M | DevelopersDigest 정리 |
> | 장점 | 초장문 컨텍스트(1M) + 매우 큰 최대 출력(384K) | DevelopersDigest 정리 |
> | 단점 | "Public beta" 성격: 운영 중 동작/품질이 바뀔 여지 | DeepSeek API Change Log |
> | 단점 | 장문/추론 모드에서 토큰을 많이 쓰는(verbose) 경향 | DevelopersDigest |
> | 단점 | 독립 벤치/런타임 측정 커버리지 부족 | BenchLM 추적 페이지 |
>
> *(전체 6개 장점 / 5개 단점 표, 원본 노트북 기준)*

```python
print(response['messages'][-1].text)  # 동일 텍스트를 plain print로 재확인
```

이 응답이 만들어지기까지 내부적으로 몇 번의 Tool 호출이 있었는지 메시지 구조로 확인합니다.

```python
# 작동 과정
rprint(response)
```

**실행 결과** (rich의 원본 컬러 dump는 검색 결과 원문까지 포함해 매우 길므로, 메시지 구조를 요약):
```
메시지 수: 7
- HumanMessage
- AIMessage | tool_calls: [current_date(), tavily_search(query='DeepSeek V4 Flash 2026 July release pros cons', ...),
                            tavily_search(query='DeepSeek V4 Flash release notes benchmark ...', ...),
                            tavily_search(query='"V4 Flash" DeepSeek 2026-07', start_date='2026-07-01', end_date='2026-08-31', ...)]
- ToolMessage  (current_date 결과)
- ToolMessage  (검색 결과 1)
- ToolMessage  (검색 결과 2)
- ToolMessage  (검색 결과 3)
- AIMessage | tool_calls: []  (최종 답변)
```

한 번의 모델 호출로 **`current_date` + `tavily_search` 3회**가 **병렬로** 호출되었습니다. Agent는 필요하다고 판단되면 여러 Tool을 동시에 호출할 수 있습니다.

### 2.4 실습: 단계별 스트리밍으로 에이전트 동작 들여다보기

`agent.stream(..., stream_mode="updates")`를 사용하면 각 스텝(모델 호출/Tool 실행)이 끝나는 시점마다 결과를 받아볼 수 있습니다.

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

```python
question = {'messages':[HumanMessage('오늘 날짜가 어떻게 되니?')]}

result = stream_agent(agent, question)
```

**실행 결과**:
```
[model]
  content: 툴을 사용해 시스템에서 제공하는 현재 날짜를 조회한 뒤, 그 값을 그대로 알려드리겠습니다.
  tool_calls: ['current_date()']

[tools]
  content: 현재 날짜는 2026-08-07 입니다!

[model]
  content: 오늘 날짜는 **2026-08-07**입니다.
```

`[model]` → `[tools]` → `[model]`의 3단계로, Agent의 내부 루프가 정확히 시각화됩니다.

### 2.5 실습: 토큰 단위 스트리밍 (`stream_utils.py`)

`stream_mode="updates"`는 각 스텝이 끝난 **시점의 결과를 한 번에** 돌려줍니다. 답변이 생성되는 과정을 **토큰 단위**로 확인하려면, 이 저장소의 `stream_utils.py`(astream_events v2 기반 헬퍼)를 사용합니다.

- `stream_print`: 토큰과 도구 호출 기록을 `print`로 출력
- `stream_with_markdown`: 토큰을 누적해 Jupyter에 마크다운으로 렌더링 (`clear_output(wait=True)` + `display(Markdown(...))` 방식)

```python
from stream_utils import stream_print, stream_with_markdown
```

```python
result = await stream_print(agent, '오늘 날짜를 확인하고, 올해가 며칠 남았는지 계산해줘. 계산기 출로 정확한 값을 구해.')
```

**실행 결과**:
```
툴 사용 계획: (1) current_date로 오늘 날짜를 확인한 뒤, (2) 올해가 윤년인지와 오늘의 연중 일수를 산출하고,
(3) calculator로 '올해 총일수 - 오늘의 연중 일수'를 계산해 올해가 끝나기까지 남은 일수를 정확히 구하겠습니다.
🔧 Tool 호출: current_date  {}
✅ Tool 결과: 현재 날짜는 2026-08-07 입니다!
중간 요약: 오늘은 2026-08-07이고, 2026년은 윤년이 아니라 총 365일입니다. 이제 8월 7일의 연중 일수(219일)를
기준으로 남은 일수 = 365 - 219를 계산기로 구하겠습니다.
🔧 Tool 호출: calculator  {'expression': '365 - 219'}
✅ Tool 결과: 146
오늘 날짜: **2026-08-07**
- 2026년은 윤년이 아니므로 총 365일
- 2026-08-07은 연중 219번째 날
- 계산(계산기): 365 - 219 = 146
따라서, 올해가 끝나기까지 남은 날(오늘 제외)은 146일입니다.
```

계산기 없이는 LLM이 산술 오류를 낼 수 있는 계산도, Tool을 통해 **정확한 값(146일)**으로 검증되었습니다.

```python
result = await stream_with_markdown(agent, 'LangChain의 create_agent가 제공하는 기능을 검색해서 마크다운 표로 정리해줘.')
```

**실행 결과** (`clear_output` 방식이라 최종 렌더링만 남습니다):

| 기능 | 무엇을 해주나 | 관련 인자/입력 |
|---|---|---|
| Tool-calling 루프 | 도구 호출이 더 없을 때까지 LLM↔도구를 반복 실행하는 그래프 생성 | `model`, `tools` |
| 메시지 상태 기반 실행 | 실행 상태에 `messages` 리스트를 누적 관리 | `{"messages": [...]}` |
| 시스템 프롬프트 주입 | 모델 호출 전 시스템 지시문을 메시지 앞에 삽입 | `system_prompt` |
| 미들웨어 확장 | 모델/도구 호출 전후 단계에서 요청/응답 수정 | `middleware` |
| 체크포인터 상태 저장 | thread_id 기준 상태 저장/복원 | `checkpointer=InMemorySaver()` |

*(Agent가 실시간 검색으로 `create_agent` 공식 문서를 찾아 정리한 표입니다 — Part 3에서 바로 이 "체크포인터"를 실습합니다.)*

---

## Part 3. 대화 메모리 — Checkpointer

### 3.1 이론: 대화 상태 관리가 필요한 이유

지금까지의 에이전트는 `invoke()`를 호출할 때마다 **빈 메시지 목록에서 시작**하므로, 직전 대화를 참조하지 못합니다. 대화를 이어가려면 **체크포인터**를 연결해 매 스텝의 상태를 저장하도록 합니다.

`create_agent`의 `checkpointer` 인자에 `InMemorySaver`를 전달하면 상태가 메모리에 보관됩니다. 보관된 상태는 `config`의 `thread_id`로 구분하므로, **같은 `thread_id`로 호출한 대화끼리만** 메시지가 이어집니다.

### 3.2 실습: `InMemorySaver`와 `thread_id`

```python
from langgraph.checkpoint.memory import InMemorySaver

memory_agent = create_agent(
    llm,
    tools=tools,
    system_prompt='주어진 도구를 사용하여 질문에 답변하세요. 앞선 대화에서 확인한 정보는 다시 묻지 말고 활용하세요.',
    checkpointer=InMemorySaver(),
)

config = {'configurable': {'thread_id': 'session-1'}}
```

```python
first = memory_agent.invoke(
    {'messages': [HumanMessage('내 이름은 김민준이고, 서울에서 데이터 분석 업무를 하고 있어.')]},
    config=config,
)
print(first['messages'][-1].text)
```

**실행 결과**:
```
알겠습니다, 김민준님. 서울에서 데이터 분석 업무를 하고 계시는군요.

지금 어떤 도움이 필요하신가요?
예: 분석 과제/지표 설계, SQL·Python 코드, 대시보드 구성, 리포트 작성, 면접/이직 준비 등
```

```python
second = memory_agent.invoke(
    {'messages': [HumanMessage('내 이름과 하는 일을 다시 알려줘.')]},
    config=config,
)
print(second['messages'][-1].text)
```

**실행 결과**: `김민준님이고, 서울에서 데이터 분석 업무를 하고 있어요.`

**같은 `thread_id`('session-1')로 호출**했기 때문에, 이전 대화에서 언급한 이름과 직업을 정확히 기억하고 있습니다.

### 3.3 실습: `get_state`로 대화 이력 조회하기

```python
state = memory_agent.get_state(config)

for message in state.values['messages']:
    print(f'{type(message).__name__}: {message.text}')
```

**실행 결과**:
```
HumanMessage: 내 이름은 김민준이고, 서울에서 데이터 분석 업무를 하고 있어.
AIMessage: 알겠습니다, 김민준님. 서울에서 데이터 분석 업무를 하고 계시는군요. ...
HumanMessage: 내 이름과 하는 일을 다시 알려줘.
AIMessage: 김민준님이고, 서울에서 데이터 분석 업무를 하고 있어요.
```

`thread_id`를 바꾸면 별도의 상태에서 대화가 시작되므로, 앞선 두 번의 대화는 참조되지 않습니다.

```python
new_config = {'configurable': {'thread_id': 'session-2'}}

third = memory_agent.invoke(
    {'messages': [HumanMessage('내 이름과 하는 일을 다시 알려줘.')]},
    config=new_config,
)
print(third['messages'][-1].text)
```

**실행 결과**:
```
지금 대화에선 당신의 이름과 하는 일(직업/역할) 정보를 받은 적이 없어서 제가 "다시" 알려줄 수가 없어요.

당신의 이름과 하는 일을 한 줄로 알려주면, 다음부터는 여기 대화 안에서 그 정보를 기준으로 다시 말해줄게요.
```

**`session-2`는 `session-1`의 대화를 전혀 알지 못합니다** — `thread_id`가 대화 세션을 완전히 격리하는 키임을 실제로 확인했습니다.

### 3.4 실습: 메모리 + 마크다운 스트리밍 결합

`stream_with_markdown`에 `config`를 함께 지정하면 `astream_events`로 그대로 전달되므로, 메모리를 연결한 에이전트에도 동일한 방식으로 사용할 수 있습니다.

```python
result = await stream_with_markdown(memory_agent, '내 이름으로 삼행시를 지어줘.', config=config)
```

**실행 결과**:
```
김: 김민준님, 데이터 속에서 답을 찾고
민: 민첩하게 가설을 세워 검증하며
준: 준비된 인사이트로 더 나은 결정을 이끕니다
```

`session-1`에서 저장된 이름("김민준")을 활용해 삼행시를 생성했습니다.

> **정리**
> - `stream_print`, `stream_with_markdown`: `astream_events`로 토큰과 도구 실행을 실시간 출력하는 `stream_utils.py`의 헬퍼
> - `InMemorySaver`: 대화 상태를 메모리에 보관하는 체크포인터 — **커널이 종료되면 상태도 사라짐**
> - `thread_id`: 하나의 대화 세션을 구분하는 키
> - `get_state`: 특정 `thread_id`에 보관된 메시지 전체를 조회하는 메서드
> - 대화를 파일이나 데이터베이스에 남기려면 `SqliteSaver`, `PostgresSaver` 같은 영속형 체크포인터를 사용합니다.

---

## Part 4. [실습 과제] 새로운 도구 3개 추가하기

> 원문 실습 지시: *"아래 조건에 맞는 에이전트를 만들어보세요: 1) 새로운 도구를 1개 이상 추가, 2) 에이전트에 적합한 시스템 프롬프트 작성, 3) 여러 도구를 조합해야 하는 질문으로 테스트"*

### 4.1 이론: MCP — Tool Calling의 표준화 (교재 p.68)

**Model Context Protocol (MCP)**은 Tool Calling을 일반화한 표준입니다. LLM의 Tool 호출 메커니즘을 **HTTP/STDIO 통신 기반의 Server/Client 관계로 표준화**하여, 크로스 플랫폼으로 동작하고 인증/미들웨어 등의 기능을 지원합니다.

- 다양한 MCP 서버 모음: [smithery.ai](https://smithery.ai/)
- 파이썬으로 MCP 서버/클라이언트를 직접 구현할 때: [FastMCP](https://gofastmcp.com/getting-started/welcome)

본 실습에서는 MCP 프로토콜을 직접 쓰지는 않지만, 아래에서 만드는 `read_file`/`write_file`/`fetch_url`/`draw_image` 같은 Tool들이 바로 MCP 서버가 표준화하려는 "LLM이 외부 세계와 상호작용하는 함수"의 예시입니다.

### 4.2 실습: `read_file` / `write_file` 도구 만들기 (보안 정책 포함)

파일 시스템 접근 Tool은 **보안이 중요**합니다. 아래 구현은 두 가지 방어 장치를 포함합니다.
1. **경로 탈출 방지**: `_resolve_within_base()`가 `os.path.realpath()`로 심볼릭 링크·`..`까지 해소한 뒤, 대상 경로가 현재 작업 디렉터리 트리 안에 있는지 검사합니다.
2. **민감 파일 차단**: `.env`, `credentials`, `id_rsa`, `.pem` 등 이름이 블록리스트에 걸리면 읽기/쓰기를 거부합니다.

```python
import os

BLOCKED_FILES = ['env', '.env', 'credentials', '.secret', 'id_rsa', '.pem']


def _resolve_within_base(file_path: str, base_dir: str | None = None) -> str | None:
    """file_path가 base_dir(기본: 현재 작업 디렉터리) 트리 안에 있으면
    정규화된 절대 경로를 반환, 벗어나면 None."""
    base = os.path.realpath(base_dir or os.getcwd())
    target = os.path.realpath(file_path)        # .. 및 심볼릭 링크까지 해소
    try:
        if os.path.commonpath([base, target]) == base:
            return target
    except ValueError:
        pass  # 서로 다른 드라이브(Windows) 등 비교 불가한 경우
    return None


def _is_blocked_name(file_path: str) -> bool:
    base = os.path.basename(file_path)
    return base in BLOCKED_FILES or any(file_path.endswith(b) for b in BLOCKED_FILES)


@tool
def read_file(file_path: str) -> str:
    """파일 내용을 읽습니다. file_path: 파일 경로"""
    if _is_blocked_name(file_path):
        return f"보안 정책에 의해 '{os.path.basename(file_path)}' 파일은 읽을 수 없습니다."
    safe = _resolve_within_base(file_path)
    if safe is None:
        return "보안 정책에 의해 현재 작업 디렉터리 밖의 파일은 읽을 수 없습니다."
    try:
        text = open(safe, encoding='utf-8').read()
        return text[:10000] + ('\n... (truncated)' if len(text) > 10000 else '')
    except Exception as e:
        return f'파일 읽기 오류: {e}'


@tool
def write_file(file_path: str, content: str) -> str:
    """파일에 내용을 작성. file_path: 경로, content: 작성할 내용"""
    if _is_blocked_name(file_path):
        return f"보안 정책에 의해 '{os.path.basename(file_path)}' 파일은 쓸 수 없습니다."
    safe = _resolve_within_base(file_path)        # makedirs 전에 검증
    if safe is None:
        return "보안 정책에 의해 현재 작업 디렉터리 밖에는 저장할 수 없습니다."
    try:
        d = os.path.dirname(safe)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(safe, 'w', encoding='utf-8') as f:
            f.write(content)
        return f'파일 작성 완료: {safe} ({len(content)}자)'
    except Exception as e:
        return f'파일 작성 오류: {e}'
```

> 💡 이 패턴은 1.5절의 "User: '(중략) report.txt에 저장해줘.' → LLM: `Function Call {Tool: Write_file, ...}`" 예시를 그대로 실제 코드로 구현한 것입니다.

### 4.3 실습: `fetch_url` 도구 만들기

```python
from langchain_community.document_loaders import WebBaseLoader

@tool
def fetch_url(url: str) -> str:
    """URL의 웹페이지 내용을 가져옵니다. url: http(s):// 웹페이지 URL"""
    if not url.startswith(('http://', 'https://')):
        return f'오류: http(s):// URL만 지원합니다. 받은 값: {url}'
    try:
        docs = WebBaseLoader(url).load()
        content = docs[0].page_content if docs else '내용을 가져올 수 없습니다.'
        return content[:5000]
    except Exception as e:
        return f'fetch_url 오류: {type(e).__name__}: {e}'
```

**실행 결과** (`fetch_url.invoke({'url': 'https://python.langchain.com/'})`):
```
USER_AGENT environment variable not set, consider setting it to identify your requests.
LangChain overview - Docs by LangChainDocumentation IndexFetch the complete documentation
index at: /llms.txtUse this file to discover all available pages before exploring further.
Skip to main contentInterrupt is coming to NYC and London this fall. Join the builders,
engineers, and teams shaping wha...
```

> `USER_AGENT` 경고는 `WebBaseLoader`가 기본 User-Agent 없이 요청을 보낼 때 나오는 안내성 경고이며, 실행 자체는 정상적으로 성공했습니다(실제 페이지 텍스트가 반환됨). 운영 환경에서는 `os.environ['USER_AGENT']`를 설정해 경고를 없애는 것을 권장합니다.

### 4.4 실습: `draw_image` 도구 만들기

`gpt-image-2` 이미지 생성 모델을 별도의 `ChatOpenAI` 인스턴스로 호출하고, 결과를 로컬 파일로 저장하는 Tool입니다.

```python
import base64
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path
from IPython.display import Image

@tool
def draw_image(prompt:str) ->str:
    """
    텍스트 프롬프트를 기반으로 AI 이미지를 생성하고 로컬에 저장합니다.

    이 도구는 사용자가 이미지 생성, 그림 그리기, 시각적 콘텐츠 제작을
    요청할 때 사용합니다. 내부적으로 GPT 이미지 생성 모델을 호출합니다.

    Args:
        prompt (str): 생성할 이미지를 설명하는 텍스트.
            구체적이고 상세할수록 결과물 품질이 향상됩니다.

    Returns:
        str: 생성된 이미지의 저장 경로를 포함한 결과 메시지.
            성공 시 "✅ Image saved: outputs/generated_images/image_YYYYMMDD_HHMMSS_mmm.png" 형식.
    """
    llm = ChatOpenAI(model="gpt-5.2", reasoning_effort='low')
    tool = {"type": "image_generation", "model": "gpt-image-2"}
    llm_with_tools = llm.bind_tools([tool])
    ai_message = llm_with_tools.invoke(prompt)

    image = next(item for item in ai_message.content_blocks if item["type"] == "image")

    kst = ZoneInfo("Asia/Seoul")
    ts = datetime.now(kst).strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
    out_dir = Path("outputs") / "generated_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"image_{ts}.png"
    out_path.write_bytes(base64.b64decode(image["base64"]))

    display(Image(filename=str(out_path)))

    return f"✅ Image saved: {out_path.as_posix()}"
```

> 💡 이미지 생성 모델은 **별도의 `llm` 지역 변수**로 새 `ChatOpenAI`를 만든 뒤 `image_generation` Tool을 바인딩합니다. Agent에 연결된 메인 `llm`(텍스트 전용)과는 별개의 모델 호출입니다. Part 5에서 이 도구가 실제로 두 차례 호출되어 고품질 한글 인포그래픽을 생성하는 것을 확인합니다.

### 4.5 실습: 7개 도구를 결합한 확장 에이전트 만들기

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 모든 도구를 하나의 리스트로 묶습니다.
tools = [
    tavily_search,
    calculator,
    current_date,
    read_file,
    write_file,
    fetch_url,
    draw_image
]

# 에이전트의 동작을 안내하는 시스템 프롬프트
system_prompt = '''당신은 유용한 AI 비서입니다. 사용자 요청에 따라 적절한 도구를 사용하여 답변을 생성합니다.
각 툴을 사용하기 전에, 툴 사용 계획에 대해 설명하고 중간 단계를 매번 간략하게 요약하세요.
사용 가능한 도구는 다음과 같습니다:
- tavily_search: 웹 검색
- calculator: 수학 계산
- current_date: 현재 날짜 확인
- read_file: 파일 내용 읽기
- write_file: 파일에 내용 작성
- fetch_url: 웹페이지 내용 가져오기
- draw_image: 텍스트 프롬프트로 이미지 생성
'''
```

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
config = {'configurable': {'thread_id': 'session-1'}}
```

두 가지 스트리밍 방식으로 같은 질문("너는무슨도구가 있니?")을 확인합니다.

```python
result = await stream_with_markdown(agent, '너는무슨도구가 있니?', config=config)
```

**실행 결과**:
```
제가 쓸 수 있는 도구는 아래 7가지예요(필요할 때만 사용합니다).

1) tavily_search: 웹 검색(최신 정보/자료 출처 찾기)
2) calculator: 수학 계산(사칙연산, 거듭제곱 등)
3) current_date: 현재 날짜 확인(yy-mm-dd 형식)
4) read_file: 파일 내용 읽기
5) write_file: 파일에 내용 작성/저장
6) fetch_url: 특정 URL 웹페이지 내용 가져오기
7) draw_image: 텍스트 프롬프트로 이미지 생성 후 저장
```

```python
result = await stream_print(agent, '너는무슨도구가 있니?', config=config)
```

**실행 결과**: (동일한 7개 도구를 stream_print의 plain-text 형식으로 재확인 — 두 스트리밍 헬퍼가 같은 정보를 다른 렌더링 방식으로 보여줌을 확인)

---

## Part 5. 복합 시나리오 — 여러 도구를 조합하는 실전 문제

### 5.1 실습: 실시간 인포그래픽 생성 (검색 + 이미지 생성 결합)

```python
result = await stream_print(agent, '''오늘 한국 프로야구 경기의 선발투수 매치업 자료를 바탕으로 인포그래픽을 생성해 줘.''', config=config)
```

**실행 결과**:
```
오늘 한국 프로야구 경기의 선발투수 매치업 자료를 바탕으로 인포그래픽을 만들기 위해, 먼저 오늘(8/7) KBO
경기 일정과 선발투수 예고 정보를 검색으로 확인한 뒤 인포그래픽을 생성하겠습니다.
🔧 Tool 호출: tavily_search  {'query': '2026-08-07 KBO 오늘 경기 선발투수 예고', 'topic': 'news', 'search_depth': 'advanced'}
✅ Tool 결과: (검색 결과: KBO가 폭염 대응으로 8/7~8/9 전 경기 취소, 8/11 재개 결정 - 관련 기사 다수)

2026-08-07(오늘) 기준으로 확인해보니 KBO가 폭염 대응으로 8/7~8/9 전 경기 취소(8/11 재개)를 결정해,
오늘은 '선발투수 매치업(선발 예고)' 자체가 성립하지 않습니다. 이 사실을 핵심으로 담은 인포그래픽을
생성했습니다.

🔧 Tool 호출: draw_image  {'prompt': 'Korean infographic ... "2026.08.07 KBO 오늘의 선발 매치업" ...
                            "폭염으로 오늘(8/7) KBO 전 경기 취소" ...'}
✅ Tool 결과: ✅ Image saved: outputs/generated_images/image_20260807_235302_977.png

- 생성 이미지 파일: `outputs/generated_images/image_20260807_235302_977.png`

원하시면 "8/11(재개일) 예정 경기 선발 매치업" 기준으로는 실제 매치업 데이터를 다시 수집해서 팀별 카드
형태 인포그래픽으로 새로 만들어드릴게요.
```

> **📷 생성된 이미지**: `draw_image`가 실제로 `gpt-image-2`를 호출해 `outputs/generated_images/image_20260807_235302_977.png`(1122×1402, 약 2MB)를 생성했습니다. **에이전트가 원래 질문("선발투수 매치업")을 그대로 수행하지 않고, 검색 결과("오늘 경기 전체 취소")에 맞춰 인포그래픽의 내용을 스스로 조정**했습니다 — 이는 고정된 Workflow가 아니라 Agent가 매 스텝마다 관찰(Observation) 결과를 반영해 계획을 수정하는 능력을 보여주는 좋은 예시입니다. 한글 텍스트 렌더링 품질도 매우 우수했습니다.

### 5.2 이론: 고정 Workflow에서 Agentic RAG로

`README.md`의 "RAG in 2026 (2)"(교재 p.55)와 "RAG의 다양한 시나리오"(수업 필기) 절에서 다룬 내용이 바로 이 실습에서 실제로 재현됩니다.

> Retrieval → Generation의 단일 워크플로우는 검색 결과에 대한 대응이 불가능합니다 (검색 결과 미비 시 추가 검색, 다중 출처 검색 등). Retrieval을 활용하는 **Agent로의 발전** — Workflow는 사라지지만, Retrieval 자체는 중요하게 남습니다.

> 기존의 RAG 파이프라인(질문 → 검색 → 답변)으로는 "여러 번의 연속 검색이 필요한 질문"(예: "삼성SDS의 클라우드 산업 경쟁사들이 27년에 진행하려고 하는 프로젝트에 대해 알려줘" → ① 경쟁사 검색 → ② 그 경쟁사의 프로젝트 검색)을 해결할 수 없습니다. → **RAG의 고정적 Workflow에서, Agentic RAG로의 변화 추세**

다음 절의 "RAG 데모 패키지 구축" 대화는 바로 이런 **여러 단계의 판단·검색·파일 생성이 연속으로 필요한 작업**을 Agent가 대화 맥락을 유지하며 수행하는 예시입니다.

### 5.3 실습: 멀티턴 대화로 RAG 데모 패키지 구축하기

새로운 `thread_id`(`session-3`)로 대화를 시작하여, 4~5턴에 걸쳐 점점 요구사항을 추가하는 시나리오입니다.

**턴 1 — 초기 요청**

```python
config = {'configurable': {'thread_id': 'session-3'}}
result = await stream_print(agent,
                            '''최신 RAG 관련 설명 샘플을 만들고 싶다. 최신 기술과 예제를 포함하여 프로그램을 생성해 줘.''',
                            config=config)
```

**실행 결과** (요약): Agent가 `tavily_search`로 "2024~2025 RAG best practices"를 검색한 뒤, **파일을 바로 쓰지 않고** Hybrid Retrieval/RRF/Multi-Query/Cross-Encoder Reranking/Contextual Compression을 포함한 `advanced_rag_sample.py` 코드를 **채팅 답변에 인라인으로 제시**하며, 로컬 실행형(A) vs API 연동형(B) 중 어떤 방향을 원하는지, 대상 문서가 무엇인지 되물었습니다. *(이 시점에는 `write_file`이 호출되지 않아 새 파일이 생성되지 않았습니다.)*

**턴 2 — 데모 패키지로 확장 요청 (⚠️ 타임아웃 발생 → 재시도)**

```python
result = await stream_print(agent,
                            '''아래 작업을 추가해 줘.
                            ---
                            1) 문서 포맷(PDF/HTML/Notion 등) 추가,
2) 한국어 NER/관계추출로 GraphRAG 품질 개선,
3) RAGAS 평가 스크립트(`evaluate.py`)까지 붙인 "데모 패키지"
로 확장해서 드릴게요. 어떤 도메인(예: 사내 규정, 기술 문서, 계약서, 논문) 문서로 샘플을 만들지 알려주시면 그에 맞춰 튜닝해드릴 수 있습니다.
.''',
                            config=config)
```

**실행 결과(최초 실행, 실패)**:
```
요청하신 "데모 패키지" 형태로 확장하기 위해 다음 순서로 진행하겠습니다.
1) 프로젝트 폴더 구조(ingest/rag/graphrag/eval) 설계 → 2) PDF/HTML/Notion 커넥터 코드 추가 →
3) 한국어 NER + 관계추출로 그래프 인덱스 생성 → 4) GraphRAG+Hybrid RAG 실행 코드 →
5) RAGAS 평가 스크립트(evaluate.py)와 샘플 QA 데이터 포함.
이제 각 파일을 생성하기 위해 write_file 도구로 패키지 파일들을 작성하겠습니다...

langchain_openai.stream_chunk_timeout fired
---------------------------------------------------------------------------
CancelledError                            Traceback (most recent call last)
...
StreamChunkTimeoutError: No streaming chunk received for 120.0s (model=gpt-5.2, chunks_received=161).
The connection may be alive at the TCP layer but is not producing content. Tune or disable via the
`stream_chunk_timeout` constructor kwarg (set to None or 0 to disable) or the
`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var.
```

> ⚠️ **실전 이슈 — `StreamChunkTimeoutError`**: 모델이 161개 청크를 스트리밍하다가 **120초 동안 추가 청크를 보내지 않아** `langchain_openai`의 클라이언트 워치독이 스트림을 강제 종료했습니다. 다중 파일 생성을 계획하는 동안(긴 `write_file` 인자를 준비하는 "생각" 단계) 모델의 응답 간격이 벌어지면서 발생한 **일시적 타임아웃**으로 추정되며, 코드 자체의 결함은 아닙니다. LangGraph의 체크포인터는 **완료되지 않은 스텝을 커밋하지 않으므로**, 대화 상태는 오류 발생 전(턴 1 종료 시점)으로 안전하게 유지되었습니다. 타임아웃 값은 `stream_chunk_timeout` 생성자 kwarg 또는 `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` 환경변수로 조정할 수 있습니다.

**재시도(동일 요청을 그대로 재전송) → 성공**

```python
# [재시도] 동일 요청 재실행 결과 (성공) — 기록용
result = await stream_print(agent,
                            '''아래 작업을 추가해 줘.
                            ---
                            1) 문서 포맷(PDF/HTML/Notion 등) 추가,
2) 한국어 NER/관계추출로 GraphRAG 품질 개선,
3) RAGAS 평가 스크립트(`evaluate.py`)까지 붙인 "데모 패키지"
로 확장해서 드릴게요. 어떤 도메인(예: 사내 규정, 기술 문서, 계약서, 논문) 문서로 샘플을 만들지 알려주시면 그에 맞춰 튜닝해드릴 수 있습니다.
.''',
                            config=config)
```

**실행 결과 (성공)**:
```
요청하신 "데모 패키지"로 확장하기 위해, (1) PDF/HTML/Notion 로더 모듈을 추가하고, (2) 한국어
NER(Transformers KLUE NER)로 엔티티를 뽑아 문장 단위 공출현+간단 패턴 기반 관계를 만들어
GraphRAG를 구현하며, (3) RAGAS로 파이프라인을 평가하는 evaluate.py까지 포함한 프로젝트 구조로
파일들을 생성하겠습니다.

🔧 Tool 호출: write_file  rag_demo/eval_data/sample_eval.csv
🔧 Tool 호출: write_file  rag_demo/README.md
🔧 Tool 호출: write_file  rag_demo/requirements.txt
🔧 Tool 호출: write_file  rag_demo/loaders/__init__.py
🔧 Tool 호출: write_file  rag_demo/loaders/pdf_loader.py
🔧 Tool 호출: write_file  rag_demo/loaders/html_loader.py
🔧 Tool 호출: write_file  rag_demo/loaders/notion_loader.py
🔧 Tool 호출: write_file  rag_demo/loaders/sample_loader.py
🔧 Tool 호출: write_file  rag_demo/storage.py
🔧 Tool 호출: write_file  rag_demo/graphrag.py
🔧 Tool 호출: write_file  rag_demo/rag_pipeline.py
🔧 Tool 호출: write_file  rag_demo/ingest.py
🔧 Tool 호출: write_file  rag_demo/run_demo.py
🔧 Tool 호출: write_file  rag_demo/evaluate.py
✅ (14개 파일 모두 "파일 작성 완료" 응답 확인)

생성된 데모 패키지 요약:

### (1) 문서 포맷(PDF/HTML/Notion 등) 추가
- loaders/pdf_loader.py: pypdf로 페이지 텍스트 추출
- loaders/html_loader.py: requests + BeautifulSoup로 본문 텍스트 추출
- loaders/notion_loader.py: Notion DB에서 title/rich_text 기반 로드
- loaders/sample_loader.py: 바로 실행 가능한 샘플 문서

### (2) 한국어 NER/관계추출로 GraphRAG 품질 개선
- graphrag.py에서 KLUE NER(klue/roberta-base-finetuned-ner)로 엔티티 추출
- 문장 내 엔티티 공출현(cooccur) 엣지 + 간단 패턴 기반 엣지(is_a, member_of 등)
- rag_pipeline.py: Hybrid 검색(BM25+Vector) → RRF fusion → GraphRAG 확장(질의 엔티티 중심 k-hop)

### (3) RAGAS 평가 스크립트(evaluate.py) 포함
- eval_data/sample_eval.csv를 읽어 파이프라인으로 답/컨텍스트 생성 후
  RAGAS metric(faithfulness, answer_relevancy, context_recall, context_precision) 계산

도메인은 3개 중 무엇으로 확정할까요? (regulation / contract / paper)
```

**14개 파일이 실제로 `rag_demo/` 디렉터리에 생성되었음을 확인**했습니다 (`write_file`의 작업 디렉터리 제한 정책이 정상적으로 현재 프로젝트 폴더 내에서 동작).

**턴 3 — 도메인별 튜닝 추가 요청**

```python
result = await stream_print(agent,
                            '''아래 작업을 추가해 줘.
                            ---
                            원하시면 다음도 같이 튜닝해 줘
- "사내 규정/계약서/논문" 중 어떤 도메인인지에 맞춰 **청킹 규칙(조항/섹션/표·캡션 유지)** 과 **BM25 vs dense 가중치**, **엔티티 stopword/용어사전**을 조정해 성능이 확 좋아집니다.  .''',
                            config=config)
```

**실행 결과 (요약)**: Agent가 먼저 `read_file`로 기존 `rag_pipeline.py`/`README.md`/`ingest.py`/`graphrag.py`를 **직접 읽어 현재 구조를 확인**한 다음, 신규 파일 7개(`domain_profiles.json`, `chunking.py`, `config_utils.py`, 도메인별 `glossary_*.json`/`entity_stopwords_*.txt` 6종)를 작성하고, 기존 5개 파일(`graphrag.py`, `rag_pipeline.py`, `ingest.py`, `run_demo.py`, `evaluate.py`)을 **수정**했습니다. 도메인별(regulation/contract/paper) 청킹 규칙, 가중 RRF(BM25 vs Dense 가중치), 엔티티 stopword/용어사전 정규화 기능이 `--domain` 옵션으로 추가되었고, 마지막으로 어떤 도메인으로 확정할지 되물었습니다.

> 💡 **`read_file` + `write_file` 조합**: Agent가 파일을 수정하기 전에 **먼저 읽어서 기존 코드 구조를 파악**한 뒤 일관된 방식으로 수정하는 모습은, 실제 코딩 에이전트(Github 이슈 해결/커밋, 1.5절 참고)의 동작 방식과 동일합니다.

**턴 4 — 짧은 후속 답변**

```python
result = await stream_print(agent, '''논문(paper)''', config=config)
```

**실행 결과**: 이전 턴에서 물었던 "도메인은 무엇으로 확정할까요?"에 대한 한 단어짜리 답변("논문")만으로도, Agent는 대화 맥락(메모리)을 통해 무엇을 묻는 질문인지 이해하고 `--domain paper` 실행 커맨드와 논문 도메인 튜닝 내역(청킹: 섹션+캡션 유지, `bm25_weight=0.45`/`dense_weight=0.55`, 엔티티 stopword/용어정규화)을 정리해 응답했습니다.

**턴 5 — 실행 방법 인포그래픽 요청**

```python
result = await stream_print(agent,
                            '''최종 RAG를 실행 하는 방법을 인포그래픽으로 출력해 줘.
                            ''',
                            config=config)
```

**실행 결과**:
```
최종 RAG 실행 방법을 "한 장짜리 인포그래픽"으로 만들기 위해, (1) 현재 데모 패키지의 실행 흐름(설치→
인제스트→실행→평가)을 핵심 단계로 요약하고, (2) 한국어 인포그래픽 레이아웃으로 구성한 뒤, (3) draw_image
툴로 이미지를 생성하겠습니다.
🔧 Tool 호출: draw_image  {'prompt': '한국어 인포그래픽 포스터, 제목: "논문(paper) 도메인 최종 RAG 실행
                           방법" ... 4단계 플로우(준비/인제스트/질의 실행/평가) ...'}
✅ Tool 결과: ✅ Image saved: outputs/generated_images/image_20260808_001058_182.png

최종 RAG 실행 방법(논문 paper 도메인)을 한 장짜리 인포그래픽으로 생성해 저장했습니다.
```

> **📷 생성된 이미지**: `outputs/generated_images/image_20260808_001058_182.png`(1672×941, 약 1.3MB) — "논문(paper) 도메인 최종 RAG 실행 방법" 4단계 플로우 인포그래픽이 정확한 한글 텍스트와 명령어로 생성되었습니다.

### 5.4 실전 이슈 정리: `StreamChunkTimeoutError`와 복구

이번 실습에서 실제로 겪은 이슈와 대응을 정리합니다.

| 상황 | 원인 | 대응 |
|---|---|---|
| 복잡한 다중 파일 생성 계획 도중 스트림 중단 | 모델이 161개 청크 이후 120초간 무응답 → 클라이언트 워치독(`stream_chunk_timeout`) 발동 | 동일 요청을 **그대로 재시도**하여 정상 완료 (체크포인터가 실패한 턴을 커밋하지 않아 대화 상태는 안전했음) |
| 장시간 실행되는 Tool(이미지 생성 등) | Bash/터미널 세션의 기본 타임아웃보다 실제 API 응답이 느릴 수 있음 | 충분히 긴 타임아웃을 설정하거나, 백그라운드 실행 + 폴링으로 대응 |
| `fetch_url` 경고 | `WebBaseLoader`가 `USER_AGENT` 미설정 시 경고 출력 | 기능에는 영향 없음. 운영 환경에서는 `USER_AGENT` 환경변수 설정 권장 |

---

## Part 6. 정리

### 6.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | Tool binding & `tool_calls` | 계산/최신정보 질문 → Tool 호출 / 일반 지식 질문 → 미호출, LLM이 상황에 맞게 판단 |
| 2 | Tool 실행 결과 재전달 | `ToolMessage`를 포함해 재호출하면 정확한 값(703,027,248)이 최종 답변에 반영됨 |
| 3 | `create_agent` 기본 루프 | `[model]→[tools]→[model]` 3단계 자동 반복, 여러 Tool 병렬 호출 확인 |
| 4 | 스트리밍(`stream_agent`, `stream_print`, `stream_with_markdown`) | 단계별/토큰별 진행 상황을 실시간으로 관찰 가능 |
| 5 | `InMemorySaver` + `thread_id` | `session-1`은 이름을 기억, `session-2`는 기억 못함 → 세션 격리 확인 |
| 6 | 신규 도구(`read_file`/`write_file`/`fetch_url`/`draw_image`) | 작업 디렉터리 제한 정상 동작, 실제 외부 호출(웹/이미지 생성) 모두 성공 |
| 7 | 실시간 인포그래픽(KBO 사례) | 검색 결과에 따라 원래 요청을 스스로 조정(경기 취소 안내로 대체) |
| 8 | RAG 데모 패키지 멀티턴 구축 | `rag_demo/` 하위 **총 23개 파일** 생성(로더 4, 리소스 6, 파이프라인/그래프/평가 스크립트 등), 도메인 튜닝까지 대화로 완성 |
| 9 | 이미지 생성 2건 | 각 1~2MB, 한글 텍스트 렌더링 품질 우수 |
| 10 | 실전 이슈 대응 | `StreamChunkTimeoutError` 발생 → 재시도로 복구, 체크포인터가 상태 안전성 보장 |

**최종 생성물**:
```
outputs/generated_images/image_20260807_235302_977.png   # KBO 오늘 경기 취소 인포그래픽
outputs/generated_images/image_20260808_001058_182.png   # 논문 도메인 RAG 실행 방법 인포그래픽
rag_demo/                                                  # 실습 대화로 생성된 RAG 데모 패키지 (23개 파일)
```

### 6.2 이론-실습 연결 정리

| 이론 (README.md) | 실습에서 확인한 것 |
|---|---|
| Tool ≈ Function, Tool Calling 흐름 (p.63~66) | `bind_tools()` → `tool_calls` 생성 → `Tool.invoke()` → `ToolMessage` → 재호출까지 전 과정을 직접 코드로 재현 |
| LangChain Tool 만들기, decorator (p.67) | `@tool`로 `calculator`/`current_date`/`read_file`/`write_file`/`fetch_url`/`draw_image` 6개 커스텀 Tool 제작 |
| Agent: Planning·Tool Executing·Observation (p.69) | `create_agent()`가 만든 `CompiledStateGraph`가 `[model]→[tools]→[model]` 루프로 정확히 동작 |
| ReAct 프롬프트 (p.70) | 시스템 프롬프트("계획을 설명하고 요약")에 따라 매 턴마다 "계획→실행→요약"이 실제로 출력됨 |
| MCP — Tool Calling 표준화 (p.68) | 직접 프로토콜을 구현하진 않았지만, 만든 Tool들이 MCP가 표준화하려는 대상과 동일한 개념임을 확인 |
| LLM Tool Calling: Taker→Giver (p.66) | `write_file` Tool이 "report.txt에 저장해줘" 시나리오를 실제 `rag_demo/` 파일 생성으로 구현 |
| RAG in 2026 (2): Agentic RAG로의 변화 (p.55) | 멀티턴 RAG 데모 패키지 구축 대화에서, Agent가 고정 파이프라인이 아니라 매 턴 계획을 세워 검색·파일 읽기·파일 쓰기를 스스로 조합함을 확인 |
| RAG의 다양한 시나리오 — 여러 번의 연속 검색 (수업 필기) | KBO 인포그래픽 실습에서, 검색 결과("경기 취소")에 따라 원래 계획(선발 매치업 인포그래픽)을 스스로 수정하는 모습으로 재확인 |

### 6.3 참고 자료

- LangChain Tools 가이드: 본 저장소 `README.md`의 "Tools in LLM (p.63)" ~ "Model Context Protocol — MCP (p.68)"
- ReAct 논문: <https://arxiv.org/abs/2210.03629>
- Tavily: <https://tavily.com/>
- Smithery (MCP 마켓): <https://smithery.ai/>
- FastMCP (파이썬 MCP 프레임워크): <https://gofastmcp.com/getting-started/welcome>
- LangGraph `create_agent` 공식 문서: <https://reference.langchain.com/python/langchain/agents/factory/create_agent>
- 이 저장소의 `stream_utils.py`: `stream_print`, `stream_with_markdown` 구현체
- 원본 실습 노트북: `[실습]_8_LangChain과_다양한_툴_연동.ipynb`

### 6.4 다음 단계

- **Lab 7** (`[실습]_7_LangChain을_이용한_SQL_데이터베이스_분석.ipynb`)에서는 Tool의 또 다른 활용인 **SQL Query Chain**(자연어 → SQL 자동 생성·실행)을 다룹니다.
- **Lab 9**에서는 Code Interpreter, Claude Code **Skills**(교재 p.72~74, `SKILL.md`) 개념을 활용해 데이터 시각화 코드를 생성·실행·개선하는 실습으로 이어집니다.
- 본 실습에서 만든 `rag_demo/` 패키지는 실제로 `cd rag_demo && pip install -U -r requirements.txt && python run_demo.py --question "..."`로 직접 실행해볼 수 있습니다 (단, Agent가 생성한 코드이므로 실행 전 검토를 권장합니다).
