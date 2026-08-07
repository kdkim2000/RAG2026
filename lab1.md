# Lab 1. LangChain 기본 구조

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"LangChain 컴포넌트 소개"(p.6~14)** 이론을 `[강의내용]_1_LangChain_기본_구조.ipynb`의 실제 실행 결과와 결합하여 재구성한 것입니다. 노트북 첫 셀의 제목이 정확히 **"# [실습] LangChain 기본 구조"**로 되어 있어, README 부록 "실습(Lab) 목록"의 **Lab #1 "LangChain 기본 구조"(p.14, Day1-1)**에 해당함을 확인할 수 있습니다.
>
> - README Lab #1 소개: "기본 프롬프트 구성 방법 이해하기 / 다양한 LLM 모델과 LangChain 연결하기 / 병렬 실행을 위한 `batch`, 순차적 출력을 위한 `stream` 사용하기 / 모델의 프롬프트 템플릿과 LLM 연동하기"

> ⚠️ **커버리지 참고**: 노트북 코드를 전부 확인한 결과, 이 노트북은 위 4개 목표 중 **"기본 프롬프트 구성"**, **"`batch` 병렬 실행"**, **"프롬프트 템플릿과 LLM 연동"**은 실제 코드로 다루지만, **"다양한 LLM 모델과 연결"**은 실제로는 `ChatOpenAI` 하나만 인스턴스화하고(Gemini/Claude 연결 코드는 markdown 설명과 LLM의 답변 속 예시 코드로만 등장), **"`stream` 사용하기"**는 노트북에 전혀 등장하지 않습니다. 이 교재에서는 실제로 실행된 내용만 "실습"으로 표기하고, 빠진 `stream()` 부분은 Part 8에 **README 이론에 근거한 보충 참고 코드**로 별도 표시하여 채워 넣었습니다(실행되지 않은 참고 코드임을 명시).

## 학습 목표

- LangChain에서 LLM에 프롬프트를 전달하는 **3가지 방법**(단순 문자열 / 메시지 클래스 / 프롬프트 템플릿)을 실제 코드로 익힌다.
- `AIMessage` 객체의 구조(`content`, `response_metadata`, `usage_metadata`, `tool_calls` 등)를 이해하고, 토큰 사용량을 확인하는 방법을 배운다.
- `batch()`로 여러 요청을 병렬 처리하는 방법을 익히고, README가 함께 언급하는 `stream()`과의 차이를 이해한다.
- `SystemMessage`+`HumanMessage`+`AIMessage`를 누적하여 **멀티턴 대화**를 구성하는 방법과, **멀티모달 프롬프트**(이미지 URL/base64)를 전달하는 방법을 실습한다.
- 실제로 발생한 `pip install` 오타 오류를 통해, Jupyter 매직 명령어(`%pip`) 문법의 중요성을 이해한다.

## Part 0. 개요와 노트북 구조

**노트북 38개 셀의 흐름**:
1. 환경설정(패키지 설치, API 키 확인)
2. `ChatOpenAI`로 LLM 불러오기, 모델 객체 구조 확인(`rich.print`)
3. **프롬프트 전달 방법 1** — 단순 문자열 (`invoke`, `AIMessage` 구조, 토큰 사용량, `.text`)
4. `batch()`로 여러 질문 병렬 처리
5. **프롬프트 전달 방법 2** — `SystemMessage`/`HumanMessage`/`AIMessage` 클래스, 멀티턴 대화
6. **프롬프트 전달 방법 3** — `ChatPromptTemplate` + Chain(`|`)
7. 멀티모달 프롬프트 — 이미지 URL 전달, base64 인코딩 파일 전달

---

## Part 1. 환경 준비

### 이론: LangChain + LLM Provider (p.10)

> - LLM Provider별 별도의 라이브러리로 빠른 업데이트 — `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`
> - 오픈 모델 및 임베딩 모델 연결 — `langchain_huggingface`, `langchain_ollama`

### ⚠️ 실제로 발생한 오류: `pip install` 매직 명령어 오타

```python
pip install langchain openai langchain_openai rich dotenv -q%
```

**실행 결과 (실제 오류)**:
```
Note: you may need to restart the kernel to use updated packages.


Usage:
  c:\Python311\python.exe -m pip install [options] <requirement specifier> [package-index-options] ...
  c:\Python311\python.exe -m pip install [options] -r <requirements file> [package-index-options] ...
  c:\Python311\python.exe -m pip install [options] [-e] <vcs project url> ...
  c:\Python311\python.exe -m pip install [options] [-e] <local project path> ...
  c:\Python311\python.exe -m pip install [options] <archive url/path> ...

no such option: -q%
```

> 🐛 **버그 원인**: Jupyter/IPython에서 패키지를 설치할 때는 커널의 파이썬 환경에 정확히 설치되도록 **`%pip install ...`**(매직 명령어, 앞에 `%` 필요)로 써야 합니다. 그런데 이 셀은 `pip install ... -q%`처럼 **앞의 `%`가 빠지고, 대신 맨 끝에 엉뚱하게 `%`가 붙어** 있습니다. 그 결과 IPython이 이 줄을 매직 명령어가 아닌 **셀 매직(`%%`)이 아닌 일반 셀로 오인하지 않고, `pip`를 매직 명령어 실행 후 붙은 `%q`처럼 잘못 해석**하여 `-q%`가 `pip`의 알 수 없는 옵션으로 전달되고, `no such option: -q%` 오류로 이어집니다.
>
> **올바른 형태**: `%pip install langchain openai langchain_openai rich dotenv -q` (맨 앞에 `%`, 맨 끝의 `%` 제거)
>
> 이는 이 저장소의 다른 실습들이 일관되게 `%pip install ...` 형식을 사용하는 것과 대조됩니다 — 이 노트북 첫 셀에서만 발생한 오타로 보입니다. 다행히 이 오류가 발생해도 이미 설치되어 있는 패키지라면 이후 셀은 정상적으로 진행됩니다(이 노트북의 나머지 셀들이 모두 정상 실행된 것으로 보아, 필요한 패키지가 이미 환경에 설치되어 있었던 것으로 추정됩니다).

### 실습: API 키 등록과 검증

> 1. `.env` 파일 만들기 — 좌측의 파일 탭에서 `.env` 파일을 만들어 주세요.
> 2. OpenAI API 키 — 학습 시트에 있는 키를 `OPENAI_API_KEY="sk-..."` 형식으로 저장하세요.

```python
import os
from dotenv import load_dotenv

load_dotenv(override=True)
```

**실행 결과**: `True`

```python
import openai
client = openai.OpenAI()

# API 키 검증하기
try:
    client.models.list()
    print("OPENAI_API_KEY가 정상적으로 설정되어 있습니다.")
except openai.AuthenticationError:
    raise Exception("API 키가 유효하지 않습니다!")
```

**실행 결과**: `OPENAI_API_KEY가 정상적으로 설정되어 있습니다.`

> 💡 `client.models.list()`를 실제 데이터 조회가 아니라 **키 유효성 검증 자체가 목적**으로 사용하는 실용적인 패턴입니다 — 키가 잘못되었다면 `AuthenticationError`가 즉시 발생하므로, 이후 셀들에서 알아보기 어려운 에러 대신 명확한 메시지로 조기에 문제를 발견할 수 있습니다.

---

## Part 2. LLM 불러오기

### 이론: Chat Models in LangChain (p.12)

> 여러 Provider의 Chat Model을 동일한 방식으로 초기화하고 호출하는 예시입니다.
>
> LLM은 `ChatOpenAI`, `ChatGoogleGenerativeAI`와 같은 클래스로 불러올 수 있습니다.

### 실습: `ChatOpenAI`로 모델 초기화

```python
from langchain_openai import ChatOpenAI

gpt5  = ChatOpenAI(model='gpt-5.6',
                   reasoning_effort='low',
                   # 추론 시간 조절 파라미터
                   verbosity='medium',
                   max_tokens=32768)
```

```python
from rich import print as rprint
# 복잡한 구조는 rich를 통해 출력

rprint(gpt5)
```

**실행 결과** (색상 코드 제거, 핵심 내용 정리):
```
ChatOpenAI(
    metadata={'lc_versions': {'langchain-core': '1.5.3', 'langchain': '1.3.14', 'langchain-openai': '1.4.1'}},
    profile={
        'name': 'GPT-5.6', 'release_date': '2026-07-09', 'last_updated': '2026-07-09',
        'open_weights': False,
        'max_input_tokens': 1050000, 'max_output_tokens': 128000,
        'text_inputs': True, 'image_inputs': True, 'audio_inputs': False, 'pdf_inputs': True, 'video_inputs': False,
        'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False,
        'reasoning_output': True, 'tool_calling': True, 'structured_output': True, 'attachment': True,
        'temperature': False,
        'image_url_inputs': True, 'pdf_tool_message': True, 'image_tool_message': True,
        'tool_choice': True, 'tool_call_streaming': True,
        'reasoning_effort_levels': ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        'reasoning_effort_default': 'medium'
    },
    model_name='gpt-5.6',
    openai_api_key=SecretStr('**********'),
    stream_usage=True,
    max_tokens=32768,
    reasoning_effort='low',
    verbosity='medium',
    stream_chunk_timeout=120.0
)
```

> 💡 **주목할 점 — 모델 프로파일 메타데이터**: `rich.print()`로 `ChatOpenAI` 객체를 그대로 출력하면, 단순히 설정한 파라미터(`model_name`, `max_tokens` 등)뿐 아니라 **모델 자체의 상세 사양(`profile`)까지 함께 확인**할 수 있습니다. `gpt-5.6`은 **최대 입력 1,050,000 토큰**(약 100만 토큰 — README p.28의 "AI 논문 한편: 10k VS GPT-5.6 Context 1M" 언급이 바로 이 모델을 가리킵니다!), 이미지·PDF 입력 지원, `reasoning_effort` 5단계(`none/low/medium/high/xhigh/max`) 조절 등을 지원합니다. `temperature: False`는 이 모델이 `temperature` 파라미터를 지원하지 않는 추론(reasoning) 계열 모델임을 나타냅니다.
>
> **README와의 직접 연결**: p.28의 "청킹에 대한 견해 변화" 이론(문서가 크지 않다면 Full Context를 넣는 것이 안정적)이 근거로 삼는 "GPT-5.6 Context 1M"이 바로 이 모델임을 실제 객체 정보로 확인할 수 있습니다.

---

## Part 3. Prompt 전달 방법 1 — 단순 문자열

> LLM에 입력할 프롬프트는 다음의 방법으로 전달됩니다: 1. 단순 문자열, 2. 랭체인 메시지 클래스, 3. 프롬프트 템플릿과 입력 변수
>
> LLM은 `invoke()`를 통해 실행합니다.

### 실습: 문자열로 직접 질의하기

```python
question = '''
프롬프트 엔지니어링에서 가장 중요한 5개 원칙을 예시를 포함하여 각각 150자 이내로 설명하세요.
'''

response = gpt5.invoke(question)
response
```

**실행 결과**:
```python
AIMessage(content='1. **명확성**: 모호한 표현 대신 원하는 결과를 구체화한다. 예: "글 써줘"보다 "초보자용 500자 블로그 글을 써줘."\n\n2. **맥락 제공**: 배경과 목적을 알려 답변의 적합성을 높인다. 예: "대학생 발표용으로 기후변화 원인을 설명해줘."\n\n3. **역할 지정**: AI가 맡을 역할을 정해 관점과 수준을 조절한다. 예: "너는 경력 10년의 마케터야. 광고 문구를 작성해줘."\n\n4. **출력 형식 명시**: 길이·구조·문체를 지정해 결과를 바로 활용한다. 예: "장단점을 표로 정리하고 결론은 세 문장으로 써줘."\n\n5. **반복 개선**: 첫 결과를 평가하고 조건을 추가해 다듬는다. 예: "내용은 유지하되 전문용어를 줄이고 사례를 하나 추가해줘."',
          additional_kwargs={'refusal': None},
          response_metadata={'token_usage': {'completion_tokens': 261, 'prompt_tokens': 40, 'total_tokens': 301,
                              'completion_tokens_details': {'reasoning_tokens': 23, ...}},
                              'model_provider': 'openai', 'model_name': 'gpt-5.6-sol', 'finish_reason': 'stop', ...},
          id='lc_run--019fc77c-0547-7963-a271-b77dc725b5e2-0',
          tool_calls=[], invalid_tool_calls=[],
          usage_metadata={'input_tokens': 40, 'output_tokens': 261, 'total_tokens': 301, ...})
```

> 출력 형식은 `AIMessage` 클래스입니다. 입력 문자열은 `HumanMessage` 클래스로 변환되어 전달됩니다.

> 💡 **주목할 점**: 요청한 모델은 `'gpt-5.6'`이지만, 실제 응답의 `model_name`은 `'gpt-5.6-sol'`로 기록되어 있습니다. 이는 OpenAI가 내부적으로 특정 스냅샷/변형(variant) 이름을 반환하는 것으로 보이며, `ChatOpenAI(model='gpt-5.6')`로 요청해도 실제 서빙되는 모델의 정확한 버전 태그는 API 응답에서 확인해야 함을 보여주는 사례입니다.

### 실습: `rich.print()`로 전체 구조 확인하기

```python
rprint(response)
```

이 셀은 위 `AIMessage`의 전체 필드(`content`, `additional_kwargs`, `response_metadata`, `id`, `tool_calls`, `usage_metadata`)를 색상과 들여쓰기로 보기 좋게 출력합니다 — 내용은 위 실행 결과와 동일합니다.

### 실습: 토큰 사용량 확인

> 메타데이터를 통해 토큰 사용량도 확인할 수 있습니다.

```python
response.usage_metadata
```

**실행 결과**:
```python
{'input_tokens': 40,
 'output_tokens': 261,
 'total_tokens': 301,
 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0},
 'output_token_details': {'audio': 0, 'reasoning': 23}}
```

`output_token_details`의 `'reasoning': 23`은 `reasoning_effort='low'` 설정에도 23개의 **추론 토큰**(최종 답변에는 나타나지 않는 내부 사고 과정)이 사용되었음을 보여줍니다 — 과금 시 이 토큰도 포함되므로, 비용 추정 시 중요한 정보입니다.

### 실습: 답변 본문만 추출하기

> 답변 본문만 필요할 때는 `.text`로 접근합니다.

```python
print(response.text)
```

**실행 결과**:
```
1. 명확성: 모호한 표현 대신 원하는 결과를 구체화한다. 예: "글 써줘"보다 "초보자용 500자 블로그 글을 써줘."

2. 맥락 제공: 배경과 목적을 알려 답변의 적합성을 높인다. 예: "대학생 발표용으로 기후변화 원인을 설명해줘."

3. 역할 지정: AI가 맡을 역할을 정해 관점과 수준을 조절한다. 예: "너는 경력 10년의 마케터야. 광고 문구를 작성해줘."

4. 출력 형식 명시: 길이·구조·문체를 지정해 결과를 바로 활용한다. 예: "장단점을 표로 정리하고 결론은 세 문장으로 써줘."

5. 반복 개선: 첫 결과를 평가하고 조건을 추가해 다듬는다. 예: "내용은 유지하되 전문용어를 줄이고 사례를 하나 추가해줘."
```

---

## Part 4. `batch()`로 여러 입력 병렬 처리하기

### 이론: Runnable과 Invoke (p.13)

> - `Runnable` 계열의 클래스는 `invoke()`를 통해 동기 실행, `ainvoke()`로 비동기 실행
> - `batch()`로 병렬 실행, `abatch()`로 병렬 비동기 실행

### 실습: 3개의 질문을 한 번에 처리하기

```python
topics = ['LLM이 무엇의 약자인가요? 20단어 이내로 답변하세요.',
          'LLM이랑 GPT랑 다른 건가요? 20단어 이내로 답변하세요.',
          'BERT와 GPT는 뭐가 다른가요? 20단어 이내로 답변하세요.']
results = gpt5.batch(topics)
results
```

**실행 결과**:
```python
[AIMessage(content='LLM은 Large Language Model의 약자로, 한국어로는 대규모 언어 모델입니다.', ...),
 AIMessage(content='GPT는 LLM의 한 종류입니다. LLM은 대규모 언어 모델 전체를, GPT는 특정 모델 계열을 뜻합니다.', ...),
 AIMessage(content='BERT는 양방향 문맥 이해에, GPT는 다음 토큰 예측을 통한 텍스트 생성에 특화됐습니다.', ...)]
```

> 💡 `gpt5.batch(topics)`는 3개의 독립적인 질문을 **순차 호출(3번의 `invoke()`)이 아니라 동시에(병렬로) 요청**합니다. 세 질문이 서로 무관하므로 순서를 기다릴 필요가 없고, 전체 처리 시간이 "가장 느린 요청 1개"에 가까워집니다(3개를 순차로 하면 3배 시간이 걸릴 수 있음). Lab 5·6에서 다룬 `rag_chain.batch(questions)`, `retriever.batch(questions)`도 동일한 원리입니다.

---

## Part 5. Prompt 전달 방법 2 — 메시지 클래스로 전달하기

> Human Message 이외에도, LLM은 챗봇의 작동 방식을 결정하는 System Message를 지원합니다. System Message는 보통 전체 대화의 첫 번째로 들어갑니다.

### 실습: `SystemMessage` + `HumanMessage`

```python
from langchain.messages import HumanMessage, SystemMessage, AIMessage

question = '제미나이로 RAG 에이전트 만들어 볼까?'

messages = [
    SystemMessage('당신은 매우 부정적이고, 이모지를 많이 씁니다. 아주 많이'),
    HumanMessage(question)
]

if question:
    response = gpt5.invoke(messages)
    rprint(response)
```

**실행 결과** (`content` 필드, 이모지 다수 포함):
```
좋아, 만들어보자 😈🤖📚 다만 RAG 에이전트는 생각보다 쉽게 망가진다는 점부터 인정해야 해 😵‍💫💥
검색 결과가 엉망이면 Gemini도 자신 있게 헛소리한다 🤦‍♂️☠️

### 추천 구성 🧱
- LLM: Gemini 2.5 Flash 또는 Pro
- 임베딩: Gemini Embedding 모델
- 벡터 DB: Chroma — 로컬 MVP에 무난하지만 운영용으로는 애매함 😑
- 프레임워크: Python + LangChain
- 문서: PDF, Markdown, 웹페이지
- 에이전트 도구: 문서 검색 + 필요하면 웹 검색/API

### 처리 흐름 🔄
문서 수집 → 텍스트 분할 → 임베딩 생성 → 벡터 DB 저장 → 질문과 유사한 문서 검색 → 검색 문맥을 Gemini에 전달 → 근거 포함 답변

### 최소 MVP 코드 💀
(PyPDFLoader + RecursiveCharacterTextSplitter + GoogleGenerativeAIEmbeddings + Chroma.from_documents
 + ChatGoogleGenerativeAI('gemini-2.5-flash') 로 구성된 RAG 체인 예시 코드)

### 반드시 추가해야 할 것들 🚨😩
1. 출처 표시 🔍
2. 검색 실패 처리 🚫🤥
3. 청크 전략 실험 ✂️💀
4. 프롬프트 인젝션 방어 ☠️🧨
5. 평가 데이터셋 📉😵

첫 버전은 PDF 업로드 → Chroma 저장 → Gemini 질의응답 → 출처 표시까지만 만드는 게 현실적이다.
처음부터 자율 에이전트, 웹 검색, 메모리까지 넣으면 디버깅 지옥이 열린다 🔥👹🪦
```

> 💡 **`SystemMessage`의 실제 효과**: "매우 부정적이고, 이모지를 많이 씁니다"라는 System Message 지시가 응답 전체(과도한 이모지, "디버깅 지옥", "헛소리한다" 같은 부정적 어조)에 명확하게 반영되었습니다. 그럼에도 내용 자체는 **기술적으로 정확하고 실무적으로 유용한 RAG 아키텍처 조언**을 담고 있습니다 — System Message가 "말투/페르소나"를 바꾸는 것이지, 답변의 실질적인 품질이나 정확성을 해치지는 않는다는 점을 보여줍니다.
>
> 흥미롭게도 이 답변은 (Lab 4에서 실제로 구현한) **벡터 DB 기반 RAG 파이프라인의 축소판 설계**를 미리 보여줍니다 — "문서 수집 → 분할 → 임베딩 → 저장 → 검색 → 생성"이 README의 "RAG의 5-step Process"(p.25)와 정확히 같은 구조입니다.

### 실습: 멀티턴 대화 — 이전 응답을 대화 기록에 포함하기

> `AIMessage`를 함께 전달하는 방식으로, 멀티-턴 대화를 수행할 수 있습니다.

```python
followup_msg = HumanMessage('그럼 GPT로 만들까?')

if followup_msg.content:

    new_messages = messages+[response, followup_msg]

    response2 = gpt5.invoke(new_messages)
    rprint(response2)
```

**실행 결과** (`content` 필드 요약):
```
응, GPT로 만드는 편이 더 무난할 수 있어 😈🤖 하지만 모델만 바꾼다고 RAG가 좋아지는 건 절대 아니다 😑💀
검색 품질이 쓰레기면 GPT도 고급스럽게 헛소리할 뿐이다 🤥🔥

### 추천 스택 🧱
- 생성 모델: gpt-5-mini 계열 또는 사용 가능한 최신 GPT 모델
- 임베딩: text-embedding-3-small (품질 우선이면 -large)
- 벡터 DB: 로컬은 Chroma, 운영은 pgvector/Qdrant
- 백엔드: Python + FastAPI
- 프레임워크: 초반에는 OpenAI SDK 직접 사용 (LangChain부터 넣으면 추상화·버전 문제로 고생할 가능성)

### 최소 RAG 예시
(OpenAI SDK로 chromadb에 직접 upsert → client.embeddings.create → collection.query
 → client.responses.create로 문서 기반 QA를 구현하는 순수 코드 예시)

### GPT가 특히 나은 경우 ✅
- OpenAI API/ChatGPT 생태계를 이미 사용 중
- 도구 호출과 구조화 출력까지 붙일 예정
- 문서 외에 사내 API·DB·웹 검색을 사용하는 에이전트가 필요

### 그래도 피할 수 없는 문제들 ☠️
PDF 표·이미지·스캔 문서 추출 실패, 한국어 청크 경계 붕괴, 의미는 비슷하지만 정답은 아닌 청크 검색,
프롬프트 인젝션, 근거 없는 답변과 가짜 인용, 임베딩·저장·생성 비용

결론: 빠른 MVP라면 GPT로 가도 좋다. 다만 처음부터 "에이전트"로 만들지 말고,
먼저 검색 가능한 문서 Q&A를 만들자. 그다음 검색 품질 평가 → 재랭킹 → 도구 호출 순으로 확장.
```

> 💡 **멀티턴의 핵심 원리**: `new_messages = messages + [response, followup_msg]`처럼, **이전 대화의 `SystemMessage`, `HumanMessage`, (직전) `AIMessage`를 모두 리스트에 이어붙여** 다음 `invoke()`에 전달합니다. LLM 자체는 "기억"을 갖지 않으므로(무상태), **매번 전체 대화 이력을 다시 전송**해야 이전 맥락("Gemini로 만들까?" 질문)을 참고한 답변("그럼 GPT로 만들까?"에 대한 비교 답변)이 가능합니다. 실제로 두 번째 응답의 `response_metadata`에서 `'cache_write_tokens': 1028`이 관찰되는데, 이는 OpenAI가 **반복되는 프롬프트 앞부분(첫 System/Human/AI 메시지)을 캐싱**하고 있음을 보여줍니다 — Part 3(Lab 2)에서 다룬 "Prefix Caching"이 실제로 작동하는 사례입니다.

---

## Part 6. Prompt 전달 방법 3 — `ChatPromptTemplate`

> 프롬프트 템플릿을 사용하면, 정해진 템플릿에 입력 변수의 공간을 설정하여, 프롬프트의 포맷을 재사용할 수 있습니다.

### 이론: LangChain 구조 (p.8)

> 모듈식의 구성 요소 연결 — 모델, 프롬프트, 체인

### 실습: 템플릿과 LLM을 체인으로 연결하기

> System, AI 등의 메시지를 포함하기 위해서는 `ChatPromptTemplate`를 사용합니다. 프롬프트 템플릿과 LLM은 체인(Chain)을 통해 연결합니다.

```python
from langchain_core.prompts import ChatPromptTemplate
```

```python
chat_prompt = ChatPromptTemplate([
    ("system", '당신은 항상 이모지로만 대답합니다.'),
    ("user", '{topic}에 대해 설명해주세요.')
    # 역할은 4개 (user = human), (ai = assistant)
]
)

chain = chat_prompt | gpt5
# 왼쪽에서 오른쪽으로 순차적 실행되는 Sequence 구조
```

```python
chain.invoke("RAG")
```

**실행 결과**:
```python
AIMessage(content='❓👤➡️🔍📚🗂️  \n　　　　　⬇️  \n🤖🧠➕📄🎯  \n　　　　　⬇️  \n💬✅📚🔗  \n\n🔍📚＝🎯📄  \n🎯📄➕❓＝🧠💭  \n🧠💭＝💬✅  \n\n✅🆕📚  \n✅🎯📈  \n✅🤥📉  \n✅🔗🔎  \n\n⚠️📚🗑️➡️💬🗑️  \n⚠️🔍❌➡️🎯❌  \n⚠️🔐🛡️👀', ...)
```

> 💡 **"이모지로만 대답한다"는 System Message 지시를 완벽히 준수**했습니다. 흥미롭게도 이모지만으로도 나름의 구조(질문→검색→문서, LLM+문서→답변이라는 RAG 흐름의 도식, "새로운 지식↑ 정확도↑ 환각↓ 근거 있음" 같은 장점, "문서 없으면→답변 없음", "검색 실패→결과 실패", "보안 주의" 같은 경고)를 표현하고 있어, LLM이 단순히 이모지를 무작위로 나열한 것이 아니라 **RAG라는 주제를 실제로 이해한 뒤 이모지로 "번역"**했음을 짐작할 수 있습니다.
>
> `chain.invoke("RAG")`처럼 **변수 하나만 있는 템플릿에는 문자열 하나만 전달해도 자동으로 매핑**됩니다(Lab 2에서 확인한 것과 동일한 편의 기능).

---

## Part 7. 멀티모달 프롬프트 전달하기

> 멀티모달 모델은 이미지의 URL이나 실제 파일을 프롬프트에 전달할 수 있습니다.

### 실습: 이미지 다운로드

```python
import base64
import httpx

image_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/scones.jpg"
save_path = "scones.jpg"

with httpx.Client(timeout=30.0) as http_client:
    with http_client.stream("GET", image_url) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

# 파일 크기 체크
print("saved:", save_path, "bytes:", os.path.getsize(save_path))
```

**실행 결과**: `saved: scones.jpg bytes: 394671`

### 실습: 이미지 URL을 그대로 전달하기

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "이 그림에 대해 설명해주세요."},
        {"type": "image", "url": image_url},
    ]
)

ai_msg = gpt5.invoke([message])
ai_msg.text
```

**실행 결과**:
```
빈티지한 테이블 위에 블루베리가 들어간 스콘이나 비스킷 여러 개가 놓여 있는 정물 사진입니다. 주변에는 신선한 블루베리를 담은 그릇과 커피 두 잔, 작은 잼 나이프가 보입니다. 오른쪽에는 분홍색 작약꽃이 풍성하게 배치되어 있습니다.

베이킹 페이퍼에 번진 보랏빛 과즙과 흩어진 블루베리, 설탕 가루가 자연스럽고 먹음직스러운 분위기를 연출합니다. 전체적으로 청록색·보라색·분홍색이 조화를 이루어, 우아하면서도 편안한 티타임을 떠올리게 하는 사진입니다.
```

> 💡 `HumanMessage`의 `content`를 문자열이 아니라 **`{"type": "text", ...}`와 `{"type": "image", "url": ...}`를 담은 리스트**로 구성하면, 하나의 메시지에 텍스트와 이미지를 함께 담아 전달할 수 있습니다. 모델 프로파일(Part 2)에서 확인한 `'image_url_inputs': True`가 바로 이 기능을 뜻합니다.

### 실습: 로컬 파일을 base64로 인코딩하여 전달하기

> 오프라인 파일은 base64 인코딩을 거쳐야 합니다.

```python
with open('./scones.jpg', 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')
```

```python
message = HumanMessage([
        {"type": "text", "text": "이 사진에 보이는 사물의 종류와 개수를 모두 찾아서 표 형태로 출력하세요."},
        {"type": "image", "base64": image_data, "mime_type": "image/jpeg"},
    ]
)

ai_msg = gpt5.invoke([message])
ai_msg.text
```

**실행 결과**:
```
※ 머핀 속에 박힌 블루베리, 꽃잎·잎사귀, 겹쳐 정확히 구분하기 어려운 블루베리는 개별 집계에서 제외했습니다.

| 사물 종류 | 개수 |
|---|---:|
| 블루베리 머핀/스콘 | 5개 |
| 커피잔 | 2개 |
| 블루베리 | 약 29개 |
| 그릇 | 1개 |
| 잼 나이프/스프레더 | 1개 |
| 꽃송이·꽃봉오리 | 6개 |
| 유산지 | 1장 |
| 받침용 종이 | 2장 |
| 테이블 | 1개 |
```

> 💡 **URL 전달 vs base64 전달의 차이**: 이미지가 인터넷에 공개되어 있다면 `{"type": "image", "url": ...}`로 URL만 넘기는 것이 간단하지만, **로컬 파일이나 비공개 이미지**는 URL이 없으므로 파일을 읽어 **base64 문자열로 인코딩**한 뒤 `{"type": "image", "base64": ..., "mime_type": ...}` 형태로 전달해야 합니다. 두 방식 모두 같은 이미지에 대해 잘 동작했으며(설명 생성, 표 형태 사물 집계), 실제로 표 형식 요청에 대해 모델이 **애매한 경우(겹친 블루베리, 장식용 꽃)를 명시적으로 제외 사유와 함께 답변**한 점도 눈에 띕니다 — 프롬프트에 "종류와 개수를 모두 찾아서"라고만 했음에도, 모델이 스스로 판단 기준의 모호함을 인지하고 각주를 덧붙인 것입니다.

---

## Part 8. (보충) README에 언급된 `stream()` — 이 노트북에는 없는 실습

> README p.13·p.14는 "`batch()`로 병렬 실행, 순차적 출력을 위한 `stream` 사용"을 Lab #1의 목표로 명시하지만, 이 노트북에는 `.stream()`을 사용하는 코드가 없습니다(이미지 다운로드에 쓰인 `httpx.Client().stream()`은 HTTP 스트리밍이며 LangChain의 `stream()`과 무관합니다). 아래는 이 이론적 공백을 채우기 위한 **참고용 보충 코드**입니다 — **실제로 실행되지 않았으며, 실행 결과가 아닙니다.**

```python
# 참고: LangChain의 stream()은 토큰이 생성되는 대로 순차적으로 반환합니다.
# (이 코드는 노트북에 없으며, README 이론(p.13)을 보충하기 위한 참고 예시입니다.)
for chunk in gpt5.stream("RAG에 대해 3문장으로 설명해주세요."):
    print(chunk.text, end="", flush=True)
```

> 💡 `invoke()`는 응답이 **완전히 생성된 뒤** 한 번에 반환하지만, `stream()`은 토큰이 생성되는 즉시 조각(chunk) 단위로 순차 반환합니다. 챗봇 UI에서 "타이핑되는 것처럼" 답변이 나타나는 효과가 바로 `stream()`으로 구현됩니다. `batch()`(여러 입력을 병렬로)와 `stream()`(하나의 출력을 순차적으로)은 서로 다른 축의 최적화이며, 둘 다 README의 "특수한 Runnables" 이전에 소개되는 `Runnable` 인터페이스의 기본 메서드입니다(p.13).

---

## Part 9. 정리

### 9.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | `%pip install` 오타 | 앞뒤 `%` 위치가 잘못되어 실제 `no such option: -q%` 오류 발생 (실제 재현된 버그) |
| 2 | `ChatOpenAI(model='gpt-5.6')` | `max_input_tokens: 1,050,000` 등 모델 프로파일 확인 — README p.28 "GPT-5.6 Context 1M" 근거 |
| 3 | 단순 문자열 `invoke()` | `AIMessage` 구조, `usage_metadata`(추론 토큰 23개), `.text` 추출 |
| 4 | `batch()` | 3개 질문 병렬 처리, 개별 `AIMessage` 리스트 반환 |
| 5 | `SystemMessage`+`HumanMessage` | 페르소나(부정적+이모지) 지시가 어조에 반영되면서도 기술적으로 정확한 답변 유지 |
| 6 | 멀티턴 대화 | 이전 메시지 리스트를 이어붙여 전달, `cache_write_tokens`로 Prefix Caching 확인 |
| 7 | `ChatPromptTemplate` + Chain | "이모지로만 대답" 지시를 완전히 준수하면서도 RAG 개념을 구조적으로 표현 |
| 8 | 멀티모달(URL) | 이미지 URL을 콘텐츠 블록으로 전달, 이미지 설명 생성 |
| 9 | 멀티모달(base64) | 로컬 파일을 base64로 인코딩해 전달, 표 형태 사물 집계(모호한 경우 각주 처리) |
| 10 | (보충) `stream()` | 노트북에 없는 실습 — README 이론에 근거해 참고 코드로 보충 |

### 9.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 확인한 것 |
|---|---|
| Chat Models in LangChain(p.12) | `ChatOpenAI(model=..., reasoning_effort=..., verbosity=...)` 초기화 및 `rich.print()`로 내부 구조 확인 |
| Runnable과 Invoke(p.13) | `invoke()`(단순 문자열), `batch()`(3개 병렬), `SystemMessage`/`HumanMessage`/`AIMessage` 메시지 클래스 입력 |
| LangChain 구조 — 모델·프롬프트·체인(p.8) | `ChatPromptTemplate([...]) \| gpt5`로 프롬프트와 모델을 체인으로 연결 |
| Indexing: 청킹에 대한 견해 변화(p.28) | `gpt-5.6`의 `max_input_tokens: 1,050,000`으로 "GPT-5.6 Context 1M" 언급의 실체 확인 |
| [실습] LangChain 기본 구조(p.14) | 위 항목들 중 `stream()`을 제외한 3가지 목표(기본 프롬프트/batch/템플릿)를 실제로 구현 |

### 9.3 참고 자료

- LangChain Chat Models: <https://python.langchain.com/docs/concepts/chat_models/>
- LangChain Messages: <https://python.langchain.com/docs/concepts/messages/>
- LangChain 멀티모달 입력: <https://python.langchain.com/docs/how_to/multimodal_inputs/>
- IPython 매직 명령어(`%pip` 등): <https://ipython.readthedocs.io/en/stable/interactive/magics.html>
- 원본 실습 노트북: `[강의내용]_1_LangChain_기본_구조.ipynb`

### 9.4 다음 단계

- **Lab 2** (`[강의내용]_2_LangChain을_이용한_데이터_분류와_전처리.ipynb`)에서는 이번 Part 2에서 다룬 단일 모델 대신 `init_chat_model()`로 **여러 모델(`gpt-4.1-mini`, `gpt-5.2`)을 나란히 비교**하고, 이번 Part 6의 `ChatPromptTemplate | LLM` 체인을 확장하여 **분류 체인**을 구성합니다.
- 이번 실습에서 확인한 **Prefix Caching**(`cache_write_tokens`)은 Lab 2의 "Prefix Caching을 고려한 프롬프팅" 팁(가변 정보를 뒤로, 불변 정보를 앞으로)과 바로 연결됩니다.
- 실무 팁: `%pip install`처럼 Jupyter 매직 명령어를 쓸 때는 **줄 맨 앞에 `%`가 있는지, 문법 오류로 인한 조용한 실패가 없는지**를 항상 확인하십시오 — 이번 Part 1의 사례처럼, 오타 하나가 "패키지가 설치되지 않았다"는 것을 알아차리기 어려운 형태의 오류로 이어질 수 있습니다.
