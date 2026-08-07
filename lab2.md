# Lab 2. LangChain을 이용한 데이터 분류와 전처리

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"LangChain 컴포넌트 소개"(p.6~14)**, **"LLM 모델의 한계"(p.21)** 이론을 `[강의내용]_2_LangChain을_이용한_데이터_분류와_전처리.ipynb`의 실제 실행 결과와 결합하여 재구성한 것입니다. 노트북 첫 셀의 제목이 정확히 **"# [실습] LangChain을 이용한 데이터 분류와 전처리"**로 되어 있어, README 부록 "실습(Lab) 목록"의 **Lab #2 "LangChain을 이용한 데이터 분류와 전처리"(p.18, Day1-2)**에 해당함을 확인할 수 있습니다.
>
> - README Lab #2 소개: "분류 프롬프트 구성하기 / LangChain 검색 모듈과 분류 체인 연결하기 / 분류 결과 평가하고 성능 개선하기"

## 학습 목표

- LCEL(`prompt | llm | parser`)의 기본 문법과, `init_chat_model()`로 여러 LLM을 교체해가며 비교하는 방법을 익힌다.
- 프롬프트 템플릿의 다중 변수 처리, 불필요한 인자 무시, **Prefix Caching을 고려한 프롬프트 설계** 원칙을 실제 코드로 확인한다.
- 실제 **Arxiv 검색 결과**에 대해 "LLM 관련 논문인지" 분류하는 체인을 만들고, 분류 결과를 필터링하여 요약 뉴스레터까지 생성하는 **2단계 파이프라인(분류 → 활용)**을 구현한다.
- 서로 다른 LLM이 "오늘 날짜"처럼 시간에 민감한 질문에 어떻게 다르게(때로는 틀리게) 답하는지 실제로 확인하여, README p.21의 "LLM 모델의 한계"를 체감한다.

## Part 0. 개요와 노트북 구조

**노트북 37개 셀의 흐름**:
1. 환경설정, `init_chat_model()`로 두 개의 LLM(`gpt-4.1-mini`, `gpt-5.2`) 비교
2. LCEL 기본: `Prompt | LLM` 체인, 다중 변수, 불필요 인자 무시
3. **[실습] 매개변수가 2개인 Prompt-LLM Chain 만들기** — 학습자 실습 + 예시 구현
4. `Prompt | LLM | Parser` 체인 (`StrOutputParser`)
5. **[실습] 검색 결과 분류 체인 만들기** — Arxiv 논문의 LLM 관련 여부 분류 (핵심 실습)
6. **[실습] LLM 최신 연구 요약 체인 만들기** — 분류 결과를 필터링하여 뉴스레터 생성

---

## Part 1. 환경 준비와 여러 LLM 모델 비교하기

### 이론: LangChain + LLM Provider (p.10), Runnable과 Invoke (p.13)

> - LLM Provider별 별도의 라이브러리로 빠른 업데이트 — `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`
> - `init_chat_model()`을 통한 API 통합 연결 — `model_provider`, `model_name` 전달

### 실습: 패키지 설치와 API 키 확인

```python
%pip install langchain langchain_openai dotenv arxiv -q
```

**실행 결과**: `Note: you may need to restart the kernel to use updated packages.`

```python
import os
from dotenv import load_dotenv
load_dotenv(override=True)
# (기본값: '.env', override=True를 통해 기존 환경 변수를 덮어쓰기 가능)

if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')
```

**실행 결과**: `OpenAI API 키 확인`

### 실습: `init_chat_model()`로 여러 모델을 런타임에 교체하기

> 랭체인에서는 아래 코드를 통해 런타임 중의 모델 수정을 지원합니다.

```python
from langchain.chat_models import init_chat_model

gpt41 = init_chat_model(
    "gpt-4.1-mini", temperature=0.3)

gpt5 = init_chat_model(
    "gpt-5.2", reasoning_effort='low')
# claude_opus = init_chat_model(
#     "claude-4.5-opus", model_provider="anthropic", temperature=0
# )

# gemini_llm = init_chat_model(
#     "gemini-3-flash-preview", model_provider="google_genai", temperature=0
# )

prompt = '모델명과 함께 자기소개를 한줄로 부탁해. 오늘은 몇월 며칠이지?'

print("GPT4.1: " + gpt41.invoke(prompt).text + "\n")
print("GPT5: " + gpt5.invoke(prompt).text + "\n")
```

**실행 결과**:
```
GPT4.1: 안녕하세요, 저는 GPT-4 모델 기반의 AI 언어 모델입니다. 오늘은 2024년 4월 27일입니다.

GPT5: 저는 OpenAI의 **ChatGPT**입니다.
오늘은 **8월 3일**입니다.
```

> ⚠️ **실제로 확인된 한계 — LLM은 "오늘 날짜"를 모른다**: 두 모델 모두 실제 실습 시점(노트북 다른 곳의 기록으로 미루어 2026년 8월경)과 **다른 날짜를 자신 있게 답변**했습니다. `gpt-4.1-mini`는 "2024년 4월 27일"(자신의 사전학습 데이터 마감 시점 근처로 추정되는 값)이라 답했고, `gpt-5.2`는 "8월 3일"이라 답했지만 연도를 언급하지 않았고 실제 실행 연도와의 일치 여부도 확인할 수 없습니다.
>
> 이는 README p.21 **"LLM 모델의 한계"**가 실제로 나타난 사례입니다.
> > 높은 수준이 항상 정확성을 의미하지는 않음 — 최신 데이터 / 내부 데이터 / 도메인 특화 데이터에 대한 질문이라면? 정확한 답변을 하지 못하는 Hallucination(환각)의 위험이 큼
>
> LLM은 시스템 시계에 접근할 수 없고, 사전학습된 데이터의 통계적 패턴으로 "그럴듯한 날짜"를 생성할 뿐입니다. 실제 서비스에서 "오늘 날짜"가 필요하다면, Lab 6·7·8에서 다루는 **Tool Calling**(현재 날짜를 반환하는 함수를 LLM에 제공)이나 **RAG**(현재 날짜 정보를 프롬프트에 직접 주입)가 필요합니다 — 이 문제가 왜 RAG/Tool이 필요한지에 대한 설득력 있는 도입부 역할을 합니다.

---

## Part 2. `Prompt | LLM` 체인 — LCEL 기초

> LangChain Expression Language(LCEL)는 랭체인에서 체인을 구성하는 문법입니다.

### 실습: 프롬프트 템플릿과 LLM 연결

```python
from langchain_core.prompts import ChatPromptTemplate

fun_chat_template = ChatPromptTemplate([
    ('user', """
### Role
당신은 영어와 한국어의 번역에 능통한 유머의 달인입니다.

### Instruction
1.  먼저, [{topic}]에 관한 영어 Pun 농담을 하나 제시하세요.
해당 농담은 한국어로 번역했을 때에서도 그 의미가 통하고 유머가 유지될 수 있어야 합니다.
만약 직역이 어렵다면, 창의적으로 각색하여 한국어 버전의 농담을 출력하세요.
- 한국어의 유사 발음, 단어의 중의적 의미, 혹은 한국의 문화적 상황 등을 활용할 수 있습니다.
2.  다음으로, 해당 농담이 영어 원어민 사용자에게 왜 재미있는지 그들의 언어적 유희 및 문화적 관점에서 한국어로 설명하세요.
""")])
```

> LCEL의 구조에서는 템플릿과 llm 모델을 설정하고, 이를 하나로 묶어 체인을 생성합니다.

```python
joke = fun_chat_template | gpt5
```

> 이후, 체인의 invoke를 실행하며 입력 포맷을 전달하면, 순서대로 체인이 실행되며 최종 결과로 연결됩니다. 입력 변수가 프롬프트 템플릿에 전달되고, 완성된 프롬프트가 LLM에 들어가는 구조입니다. 입력 포맷은 Dict 형식으로 전달합니다.

```python
response = joke.invoke({'topic':'eggs'})
# 매개변수가 1개일 때는 joke.invoke('eggs') 도 가능
print(response.text)
```

**실행 결과**:
```
1) Eggs 관련 영어 Pun + 한국어 버전

- EN: Why did the egg hide? Because it was a little chicken.
- KR(의미 살린 번역/각색): 달걀이 왜 숨었을까? 자기가 "치킨(겁쟁이)"인 걸 알았거든.

(한국어에선 "치킨"이 실제 닭고기이기도 하고, 속어로 "겁쟁이"라는 뜻도 있어서 말장난이 그대로 살아납니다.)

---

2) 왜 영어 원어민에게 재미있는지 (한국어 설명)

이 농담의 핵심은 "chicken"의 중의성입니다.
- 영어에서 chicken은 원래 "닭"을 뜻하지만, 동시에 일상적으로 겁이 많은 사람(겁쟁이)을 놀릴 때 You're a chicken.처럼 씁니다.
- "egg(달걀)"은 "chicken(닭)"과 생물학적으로 이어지는 존재라서, 달걀이 "I'm a little chicken"이라고 말하면 겹친 의미가 한 번에 터집니다.
- 문화적으로도 영어권(특히 미국)에서는 "chicken(겁쟁이)"가 아주 흔한 표현이라, 듣는 순간 즉각적으로 중의성을 캐치하고 가볍게 웃을 수 있는 "dad joke(아재개그)"로 잘 먹힙니다.
```

### 실습: 프롬프트에 없는 변수는 무시됨

```python
response = joke.invoke({'topic':'pigeon', 'foo':'bar'})
# 프롬프트에 포함되어 있지 않은 매개변수는 무시
print(response.text)
```

**실행 결과** (일부):
```
1) 영어 Pun 농담 + 한국어 버전(의미/유머 유지 각색)

- EN: Why don't pigeons use email? Because they prefer coo-mail.
- KR(각색): 비둘기는 왜 이메일을 안 쓸까? '구구메일'이 더 좋거든.
  (비둘기 울음소리 "coo(구/쿠)" + mail → coo-mail / 구구+메일)

---

2) 영어 원어민에게 왜 재미있는지(한국어 설명)

이 농담의 핵심은 말장난(pun)입니다.
- 영어에서 비둘기 울음소리를 "coo"(쿠- 하는 소리)라고 표현합니다.
- 그리고 "email"과 비슷한 형태로 "coo-mail"이라는 가짜 단어를 만들어, 마치 비둘기들이 이메일 대신 자신들만의 메일 서비스를 쓰는 것처럼 보이게 합니다.
- 동시에 "pigeon"은 역사적으로 carrier pigeon(전서구)처럼 편지를 나르던 이미지가 있어, "메일(mail)"과 연결되는 문화적 배경이 자연스럽습니다.

한국어 버전의 "구구메일"은 영어의 "coo"에 해당하는 비둘기 의성어 "구구"를 붙여 같은 구조의 말장난을 재현한 각색입니다.
```

`{'topic':'pigeon', 'foo':'bar'}`에서 `foo`는 프롬프트 템플릿(`{topic}`만 사용)에 없는 변수이므로 **조용히 무시**되고 에러가 발생하지 않았습니다.

### 실습: 체인이 LLM에 실제로 전달하는 프롬프트 확인하기

```python
# 체인이 LLM에 전달하는 실체
fun_chat_template.invoke({'topic':'eggs'}).messages
```

**실행 결과**:
```python
[HumanMessage(content='\n### Role\n당신은 영어와 한국어의 번역에 능통한 유머의 달인입니다.\n\n### Instruction\n1.  먼저, [eggs]에 관한 영어 Pun 농담을 하나 제시하세요.\n해당 농담은 한국어로 번역했을 때에서도 그 의미가 통하고 유머가 유지될 수 있어야 합니다.\n만약 직역이 어렵다면, 창의적으로 각색하여 한국어 버전의 농담을 출력하세요.\n- 한국어의 유사 발음, 단어의 중의적 의미, 혹은 한국의 문화적 상황 등을 활용할 수 있습니다.\n2.  다음으로, 해당 농담이 영어 원어민 사용자에게 왜 재미있는지 그들의 언어적 유희 및 문화적 관점에서 한국어로 설명하세요.\n', additional_kwargs={}, response_metadata={})]
```

> 💡 `ChatPromptTemplate.invoke(...)`만 실행하면(LLM을 거치지 않고) `{topic}`이 실제 값(`eggs`)으로 채워진 **`HumanMessage` 객체 리스트**를 확인할 수 있습니다. 프롬프트가 실제로 LLM에 어떤 형태로 도달하는지 디버깅할 때 유용한 패턴입니다.

---

## Part 3. [실습] 매개변수가 2개인 Prompt-LLM Chain 만들기

> 임의의 `ChatPromptTemplate`를 만들고, 2개의 매개변수를 받도록 구성하여 체인을 만들고 실행하세요.

### 실습: 예시 1 — 한국어/외국어 병렬 문장 생성

```python
# 아래 LLM을 사용하세요!
gpt5 = init_chat_model(
    "gpt-5.6", reasoning_effort='low')
```

```python
prompt = ChatPromptTemplate(
    [
        ('system','''주어진 주제로, 10문장 길이의 짧은 글을 작성하세요.
한국어 문장과, 그 문장을 다음 언어로 번역한 문장을 번갈아 가며 출력하세요.'''),
        ('human','''
주제: {topic}
언어: {language}
''')
    ]
)
# System, Human 구조, (매개변수 2개는 자유로운 위치에)
# Prefix Caching을 고려한 프롬프팅
# 불변 패턴은 앞부분에, 가변 패턴은 뒷부분에 넣는 프롬프트 권장
# # 좋지 않은 패턴
# prompt = ChatPromptTemplate(
#     [
#         ('system','''{topic}에 대해, 10문장 길이의 짧은 글을 작성하세요.
# 한국어 문장과, 그 문장을 다음 언어로 번역한 문장을 번갈아 가며 출력하세요.'''),
#         ('human','''
# 언어: {language}
# ''')
#     ]
# )
```

> 💡 **참고할 만한 내용 — Prefix Caching을 고려한 프롬프팅**: 주석으로 남겨진 "좋지 않은 패턴"과 실제 사용된 패턴을 비교하면, **가변 정보(`{topic}`)를 시스템 프롬프트 앞부분이 아니라 사람(human) 메시지 뒷부분에 두는 것**이 권장됨을 알 수 있습니다. 많은 LLM Provider(OpenAI 포함)는 **동일한 프롬프트 접두사(prefix)가 반복될 경우 캐싱하여 비용과 지연시간을 절감**합니다. 시스템 프롬프트처럼 매 호출마다 바뀌지 않는 "불변 패턴"을 앞부분에 고정하고, 매번 달라지는 "가변 패턴"(주제, 사용자 입력 등)을 뒤로 보내면 캐시 적중률이 높아집니다. 이는 README에는 명시적으로 언급되지 않지만, 실무에서 프롬프트를 설계할 때 비용 최적화 관점에서 매우 중요한 원칙입니다.

```python
chain = prompt | gpt5
```

```python
result = chain.invoke({'topic':'타코의 종류', 'language':'스페인어'})
print(result.text)
```

**실행 결과**:
```
타코에는 사용되는 재료와 조리법에 따라 다양한 종류가 있습니다.
Hay diversos tipos de tacos según los ingredientes y la forma de preparación.
타코 알 파스토르는 양념한 돼지고기와 파인애플을 넣어 만듭니다.
Los tacos al pastor se preparan con carne de cerdo adobada y piña.
카르니타스 타코는 부드럽게 익힌 돼지고기가 특징입니다.
Los tacos de carnitas se caracterizan por su carne de cerdo cocida hasta quedar tierna.
생선 타코에는 튀기거나 구운 생선과 신선한 채소가 들어갑니다.
Los tacos de pescado llevan pescado frito o a la parrilla y verduras frescas.
채식 타코는 콩, 버섯, 아보카도 같은 재료로 맛을 냅니다.
Los tacos vegetarianos se elaboran con ingredientes como frijoles, champiñones y aguacate.
```

### 실습: 예시 2 — 두 캐릭터의 만남 대화 생성

```python
prompt = ChatPromptTemplate(
    [
        ('system', '당신은 재미있고 교훈적인 이야기를 씁니다.'),
        ('user', '{A}와 {B}가 만났을 때의 대화를 써 주세요.')
    ])
chain = prompt | gpt5
response = chain.invoke({'A':'햄릿', 'B':'슈퍼마리오'})
print(response.text)
```

**실행 결과** (전문 — "햄릿과 슈퍼마리오: '할 것인가, 점프할 것인가'"):

> 어두운 엘시노어 성. 햄릿이 해골을 들고 깊은 생각에 잠겨 있었다. 그때 천장에서 초록색 파이프 하나가 내려오더니, 빨간 모자를 쓴 슈퍼마리오가 튀어나왔다.
>
> **햄릿:** 할 것인가, 말 것인가. 그것이 문제로다.
>
> **마리오:** 일단 파이프에서 나올 것인가, 다시 들어갈 것인가. 그것도 문제네!
>
> **햄릿:** 그대는 누구인가? 복장이 지나치게 명랑하군.
>
> **마리오:** 나는 마리오! 버섯 왕국에서 온 배관공이지. 공주를 구하러 가다가 길을 잘못 들었어. 여긴 쿠파 성이 아닌가?
>
> **햄릿:** 이곳은 덴마크의 엘시노어 성이다. 괴물보다 음모가 많고, 용암보다 소문이 뜨거운 곳이지.
>
> **마리오:** 그럼 왕에게 길을 물어보면 되잖아?
>
> **햄릿:** 왕은 믿을 수 없다.
>
> ... *(중략 — 두 인물이 "신중함"과 "행동력"에 대해 철학적 대화를 이어가며, 유령의 등장과 마리오의 점프 개그로 이어짐)* ...
>
> **햄릿:** 할 것인가, 말 것인가? 이제 답을 알겠군.
> **생각한 뒤, 해 보는 것이다.**

> 💡 두 실습 예시 모두 **동일한 2-변수 LCEL 패턴**(`ChatPromptTemplate([system, human]) | llm`)을 따르지만, 하나는 정보 전달형(번역 병기), 다른 하나는 창작형(대화 생성)으로 응용 범위가 다름을 보여줍니다.

---

## Part 4. `Prompt | LLM | Parser` 체인

> LCEL의 체인에는 파서(Parser)를 추가할 수 있습니다. 파서는 출력 형식을 변환합니다.
>
> **StrOutputParser**: 출력 결과를 String 형식으로 변환합니다.

### 실습: 파서 추가하기

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

recipe_template=ChatPromptTemplate([
    ('system','당신은 전세계의 조리법을 아는 쉐프입니다.'),
    ('user','''저는 다음의 재료를 이용한 환상적인 요리를 만들고 싶습니다.

레시피와 함께, 고객의 시선을 사로잡을 수 있는 추천사도 작성해 주세요.
---
[재료]: {ingredient}''')
])
```

```python
recipe_chain = recipe_template | gpt5 | parser
response = recipe_chain.invoke({'ingredient':'커피, 연두부, 에너지바, 바나나'})
print(response)
```

**실행 결과**:
```
## 커피 향 가득한 바나나 연두부 티라미수 파르페

부드러운 연두부와 바나나로 크림을 만들고, 진한 커피를 머금은 에너지바를 층층이 쌓은 노오븐 디저트입니다.
고소함과 달콤함, 쌉싸름한 커피 향을 한 컵에 담았습니다.

### 재료 — 2인분
- 연두부 300g
- 잘 익은 바나나 2개
- 에너지바 2개 (견과류·귀리 계열이 특히 잘 어울립니다.)
- 진하게 내린 커피 또는 에스프레소 80ml, 완전히 식힌 것

### 만드는 법
1. 연두부의 물기를 제거합니다. (체에 올려 10분 정도)
2. 바나나 연두부 크림을 만듭니다. (연두부 + 바나나 1½개를 믹서에 갈기)
3. 커피 크럼블을 준비합니다. (에너지바에 식힌 커피를 촉촉하게 적시기)
4. 파르페를 조립합니다. (커피 에너지바 → 크림 → 바나나 슬라이스 순으로 층 쌓기, 2회 반복)
5. 차갑게 굳힙니다. (냉장고 30분~1시간)
6. 마무리합니다. (에너지바 부스러기, 코코아가루/시나몬)

## 고객의 시선을 사로잡는 추천사
> "티라미수의 우아함을 한층 가볍고 색다르게!"
> 진한 커피를 머금은 에너지바 위로 구름처럼 부드러운 바나나 연두부 크림이 겹겹이 쌓입니다.
> 첫입은 달콤하고, 끝맛은 쌉싸름하며, 은은한 고소함까지 남는 매력적인 노오븐 디저트.
```

> `recipe_chain`에 `| parser`가 추가되면서, `AIMessage` 객체가 아니라 **순수 문자열**이 반환됩니다 — `print(response)`가 `response.text`가 아니라 `response`만으로 바로 동작하는 이유입니다.

---

## Part 5. [실습] 검색 결과 분류 체인 만들기 — 이 Lab의 핵심 실습

> 다음은 Arxiv의 최신 논문을 검색하는 함수입니다. 해당 논문들이 LLM 관련 논문인지 분류하는 체인을 만들고, 실행하여 결과를 비교하세요. 함수의 결과물로 다양한 값들이 있으므로, 값들 중 필요한 값만 입력받는 체인을 만들고 실행하세요.

이 실습이 바로 README Lab #2의 **"LangChain 검색 모듈과 분류 체인 연결하기"**에 정확히 대응합니다.

### 실습: Arxiv 검색 함수

```python
import arxiv
from typing import List, Dict, Optional

def get_arxiv_papers(query: Optional[str] = None, N: int = 10) -> List[Dict]:
    """
    arXiv에서 논문 리스트를 가져오는 함수

    Parameters:
    -----------
    query : str, optional
        검색어
    N : int, default=10
        가져올 논문 개수

    Returns:
    --------
    List[Dict] : 논문 정보를 담은 딕셔너리 리스트
    """


    search_query = query

    # arxiv 클라이언트 생성
    client = arxiv.Client()

    # 검색 객체 생성
    search = arxiv.Search(
        query=search_query,
        max_results=N,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 제출일 기준 정렬
        sort_order=arxiv.SortOrder.Descending  # 최신순
    )

    # 결과를 저장할 리스트
    papers = []

    # 검색 실행 (새로운 API 사용)
    for result in client.results(search):
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published': result.published.strftime('%Y-%m-%d %H:%M:%S'),
            'updated': result.updated.strftime('%Y-%m-%d %H:%M:%S'),
            'arxiv_id': result.entry_id.split('/')[-1],  # arXiv ID 추출
            'pdf_url': result.pdf_url,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'comment': result.comment,
            'journal_ref': result.journal_ref
        }
        papers.append(paper_info)

    return papers

query = 'Security'
print(f"\n\n=== 검색어 `{query}` 로 검색한 최근 논문 ===")
security_papers = get_arxiv_papers(query=query, N=5)
print(f"총 {len(security_papers)}개의 논문을 가져왔습니다.")
print('\n'.join([paper['title'] for paper in security_papers]))

# 임의의 검색어로 검색하려면 query를 바꿔 다시 호출하세요.
```

**실행 결과** (arXiv 실시간 검색, 실제 API 호출):
```

=== 검색어 `Security` 로 검색한 최근 논문 ===
총 5개의 논문을 가져왔습니다.
CWEEP: A Lexical Static Analysis Framework for CWE Early Prevention
Beyond Resilience: Antifragility in Critical Infrastructure Cybersecurity
From Code Review to Code Critique: Intent, Drift, and Spotlight for AI-Generated Diffs at Scale
Bending the Curve: Operational Cyber Epidemiology for Ransomware
AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair
```

> 💡 이 셀은 실시간 `arxiv` 공식 API(`arxiv` 파이썬 패키지)를 호출합니다 — README p.29 "LangChain Document Loader"의 "Arxiv ... 등의 API 로더"가 바로 이런 외부 API 연동을 가리킵니다. 검색 결과는 **실행 시점마다 달라질 수 있으므로**(매일 새 논문이 게시됨), 노트북은 이후 셀에서 특정 시점(2026-07 하순)의 검색 결과를 **하드코딩된 데이터로 고정**하여 재현성을 확보합니다.

### 실습: 재현 가능한 고정 데이터셋 (5개 논문 상세정보)

```python
security_papers = [
    {
        'title': 'CWEEP: A Lexical Static Analysis Framework for CWE Early Prevention',
        'authors': ['Bryan Kwan', 'Benjamin Tan'],
        'summary': (
            'This paper presents CWEEP, a lexical static-analysis framework '
            'for detecting security weaknesses in register-transfer-level hardware designs. '
            'CWEEP identifies vulnerable RTL code locations and suggests repairs without '
            'requiring a complete security specification. The evaluation includes an '
            'LLM-generated dataset of 3,874 buggy hardware modules, but the proposed '
            'security-analysis method itself is not based on a large language model.'
        ),
        'published': '2026-07-31 16:35:13', 'updated': '2026-07-31 16:35:13',
        'arxiv_id': '2607.29604', 'pdf_url': 'https://arxiv.org/pdf/2607.29604',
        'categories': ['cs.CR'], 'primary_category': 'cs.CR',
        'comment': '12 pages, 9 figures', 'journal_ref': None
    },
    {
        'title': 'AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair',
        'authors': ['Michael Fu', 'Qiyue Mei', 'Patanamon Thongtanunam', 'Kla Tantithamthavorn'],
        'summary': (
            'This paper presents AgenticRepair, an LLM-based multi-agent framework '
            'for automatically repairing software vulnerabilities. Three specialized '
            'LLM subagents collect code-structure, runtime-execution, and commit-history '
            'context, which is passed to a repair agent for patch generation. '
            'On 300 real-world SEC-Bench cases, AgenticRepair achieves a 73 percent '
            'successful repair rate with sanitizer-based patch verification.'
        ),
        'published': '2026-07-31 13:42:51', 'updated': '2026-07-31 13:42:51',
        'arxiv_id': '2607.29422', 'pdf_url': 'https://arxiv.org/pdf/2607.29422',
        'categories': ['cs.SE', 'cs.AI', 'cs.CR'], 'primary_category': 'cs.SE',
        'comment': 'Under Review at IEEE TSE', 'journal_ref': None
    },
    {
        'title': 'SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response',
        'authors': ['Lehan Wang', 'Boli Chen', 'Ruixue Ding', 'Pengjun Xie', 'Jinwei Huang',
                     'Zhendong Liu', 'Shuo Wang', 'Tao Lei', 'Xin Ouyang', 'Xiaomeng Li'],
        'summary': (
            'This paper introduces SecRespond, a benchmark for evaluating LLM agents '
            'on real-world post-compromise incident-response tasks. Agents analyze '
            'forensic disk snapshots, alerts, vulnerability scans, and system baselines '
            'to identify intrusions and generate remediation plans. The authors evaluate '
            '23 frontier LLMs across 10 compromised cloud-host environments and find '
            'that current agents struggle with silent intrusions and verified remediation.'
        ),
        'published': '2026-07-29 11:32:23', 'updated': '2026-07-29 11:32:23',
        'arxiv_id': '2607.26791', 'pdf_url': 'https://arxiv.org/pdf/2607.26791',
        'categories': ['cs.CR', 'cs.AI', 'cs.CL'], 'primary_category': 'cs.CR',
        'comment': None, 'journal_ref': None
    },
    {
        'title': 'ALIBI: Adaptive Agentic Attacks on LLM-Based Vulnerability Detectors via Adversarial Code Comments',
        'authors': ['Zixuan Wu', 'Cristina Nita-Rotaru'],
        'summary': (
            'This paper studies attacks against LLM-based vulnerability detectors. '
            'It introduces ALIBI, an adaptive black-box attack framework in which '
            'a coding agent inserts vulnerabilities and adversarial source-code comments '
            'designed to manipulate the detector reasoning. Across 125 real-world '
            'vulnerabilities, attack success rates exceed 90 percent for all evaluated '
            'detectors. Architectural isolation and comment sanitization are more '
            'effective than prompt-level defenses.'
        ),
        'published': '2026-07-27 18:13:28', 'updated': '2026-07-27 18:13:28',
        'arxiv_id': '2607.24964', 'pdf_url': 'https://arxiv.org/pdf/2607.24964',
        'categories': ['cs.CR'], 'primary_category': 'cs.CR',
        'comment': None, 'journal_ref': None
    },
    {
        'title': 'Just Testing, Move Along: Evasion of LLM-based System Log Interpretation by Prompt Injection',
        'authors': ['Max Landauer', 'Florian Skopik', 'Markus Wurzenberger', 'Franciszek Górski', 'Mateusz Krzysztoń'],
        'summary': (
            'This paper evaluates prompt-injection attacks against LLM-based system-log '
            'analysis in Security Operations Center workflows. Attackers insert malicious '
            'instructions into log entries so that LLMs interpret genuine indicators '
            'of compromise as benign activity. Experiments with multiple state-of-the-art '
            'LLMs show that optimized log injections can successfully evade detection. '
            'The generated explanations may nevertheless provide signals for identifying '
            'the manipulation.'
        ),
        'published': '2026-07-27 08:59:00', 'updated': '2026-07-27 08:59:00',
        'arxiv_id': '2607.24174', 'pdf_url': 'https://arxiv.org/pdf/2607.24174',
        'categories': ['cs.CR'], 'primary_category': 'cs.CR',
        'comment': None, 'journal_ref': None
    }
]

print(f"총 {len(security_papers)}개의 논문을 가져왔습니다.")
print('\n'.join(paper['title'] for paper in security_papers))
```

**실행 결과**:
```
총 5개의 논문을 가져왔습니다.
CWEEP: A Lexical Static Analysis Framework for CWE Early Prevention
AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair
SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response
ALIBI: Adaptive Agentic Attacks on LLM-Based Vulnerability Detectors via Adversarial Code Comments
Just Testing, Move Along: Evasion of LLM-based System Log Interpretation by Prompt Injection
```

> 💡 참고로, 위 5개 논문 중 1개("CWEEP")는 **하드웨어(RTL) 보안 분석 논문으로 LLM과 무관**하고, 나머지 4개는 모두 **LLM을 명시적으로 활용하는 보안 연구**입니다. 이 5개를 검색어 `'Security'`로 함께 가져왔다는 것 자체가, README p.32 "Searching: 쿼리와 적합한 데이터 검색하기"의 **키워드 기반(Lexical) 검색이 항상 정확하지는 않다**는 점을 보여줍니다 — "Security"라는 키워드만으로는 "LLM 관련"인지 아닌지를 구분할 수 없으므로, 이 다섯 편을 다시 한 번 **LLM으로 분류**해야 하는 이유가 명확해집니다.

### 실습: 분류 프롬프트와 분류 체인 구성

```python
# security_papers
# LLM 관련 논문이면 '분류 결과: LLM'을 뒤에 출력
# 관련 논문이 아니면 '분류 결과: Not LLM'을 뒤에 출력

# prompt | llm | parser

classify_prompt = ChatPromptTemplate(
    [# 2개의 매개변수를 받아 분류
        ('system','''다음의 논문과 LLM의 관련성에 대해 200자 이내로 설명하세요.
LLM 관련 논문이면 '분류 결과: LLM'을 마지막에 출력하세요.
관련 논문이 아니면 '분류 결과: Not LLM'을 마지막에 출력하세요.
'''),
        ('human', '''
논문 제목: {title}
논문 요약: {summary}
''')
    ]
)

classify_chain = classify_prompt | gpt5 | parser
```

> 💡 **README Lab #2 "분류 프롬프트 구성하기"의 실제 구현**입니다. `classify_prompt`는 `{title}`, `{summary}` **2개의 매개변수**만 받습니다 — Part 5 도입부의 실습 지시("함수의 결과물로 다양한 값들이 있으므로, 값들 중 필요한 값만 입력받는 체인을 만들고 실행하세요")를 그대로 만족합니다. `security_papers`의 각 딕셔너리에는 `authors`, `pdf_url`, `categories` 등 총 10개 필드가 있지만, 분류에는 `title`과 `summary` 2개만으로 충분하다고 판단한 것입니다.

```python
classification_result = classify_chain.batch(security_papers)
classification_result
```

**실행 결과**:
```python
['LLM 생성 데이터셋을 평가에 활용했지만, 핵심 기법은 RTL 하드웨어 취약점을 탐지·수정하는 어휘 기반 정적 분석으로 LLM 기반 연구가 아닙니다.\n분류 결과: Not LLM',
 'LLM 기반 다중 에이전트가 코드 구조·실행·커밋 이력을 분석해 취약점 패치를 생성하고 검증하는 연구입니다.\n분류 결과: LLM',
 'SecRespond는 침해 후 사고 대응에서 LLM 에이전트의 포렌식 분석, 침입 식별, 복구 계획 수립 능력을 평가하는 벤치마크다. 23개 최신 LLM의 한계도 분석한다.\n분류 결과: LLM',
 'LLM 기반 취약점 탐지기를 적대적 코드 주석으로 교란하는 에이전트형 공격과 방어 기법을 연구한 논문입니다.\n분류 결과: LLM',
 'LLM 기반 보안 로그 분석 시스템을 대상으로 프롬프트 주입 공격의 탐지 우회 가능성과 대응 단서를 평가한 연구입니다.\n분류 결과: LLM']
```

**분류 결과: 5편 중 4편 "LLM", 1편 "Not LLM"** — `classify_chain.batch(security_papers)`는 `security_papers`(딕셔너리 리스트)의 **각 딕셔너리를 프롬프트의 필요한 변수(`title`, `summary`)에 자동으로 매핑**하여 5건을 한 번에 병렬 분류합니다. `security_papers`에는 `authors`, `pdf_url` 등 여러 필드가 더 있지만, `classify_prompt`가 `{title}`/`{summary}`만 요구하므로 나머지는 자동으로 무시됩니다(Part 2에서 확인한 "필요 없는 인자는 무시" 동작이 여기서도 그대로 적용).

> ⚠️ **README Lab #2 "분류 결과 평가하고 성능 개선하기"와 관련된 실무적 관찰**: 이 분류는 프롬프트에 `"'분류 결과: LLM'을 마지막에 출력하세요"`라는 **자유 형식 텍스트 규칙**만으로 이루어졌습니다. Part 2·3(Lab 3)에서 배운 Pydantic/`with_structured_output()`으로 `Literal['LLM', 'Not LLM']` 필드를 강제했다면, `'분류 결과: LLM' in classification_result[i]`처럼 **문자열 포함 여부로 파싱해야 하는 취약한 방식** 대신, `result.label == 'LLM'`처럼 안전하게 분류 결과를 다룰 수 있었을 것입니다. 이것이 바로 다음 Part의 필터링 코드에서 실제로 어떻게 쓰이는지 확인해봅니다.

---

## Part 6. [실습] LLM 최신 연구 요약 체인 만들기 — 분류 결과의 활용

> 분류 결과를 바탕으로, LLM 관련 논문만 모아 요약할 수 있습니다. 적절한 요약 프롬프트를 생성하여, 이전 실습의 결과 중 LLM에 해당하는 결과들만을 모으세요.

이 Part가 README Lab #2의 **"분류 결과 평가하고 성능 개선하기"**를 "분류 결과를 실제로 다음 단계에 활용"하는 형태로 구현합니다 — 분류는 그 자체가 목적이 아니라, **후속 작업(요약)의 입력을 필터링하는 전처리 단계**로 쓰입니다.

### 실습: 분류 결과로 LLM 관련 논문만 필터링하기

```python
LLM_documents=[]

# LLM 분류 조건 만족시, LLM_documents에 정보 저장
# 정보: 문자열 형식 (논문 제목, 날짜, 저자, PDF 주소, 요약)
# LLM_documents : 문자열 리스트
for i in range(len(classification_result)):
    if '분류 결과: LLM' in classification_result[i]:
        paper_info = f"""
논문 제목: {security_papers[i]['title']}
게시 날짜: {security_papers[i]['published']}
저자: {security_papers[i]['authors']}
URL: {security_papers[i]['pdf_url']}
요약: {security_papers[i]['summary']}
"""
        LLM_documents.append(paper_info)

# LLM_documents

context = '\n'.join(LLM_documents)
print(context)
```

**실행 결과** (4편의 LLM 관련 논문만 필터링됨 — "CWEEP" 논문은 제외됨):
```

논문 제목: AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair
게시 날짜: 2026-07-31 13:42:51
저자: ['Michael Fu', 'Qiyue Mei', 'Patanamon Thongtanunam', 'Kla Tantithamthavorn']
URL: https://arxiv.org/pdf/2607.29422
요약: This paper presents AgenticRepair, an LLM-based multi-agent framework for automatically repairing software vulnerabilities. Three specialized LLM subagents collect code-structure, runtime-execution, and commit-history context, which is passed to a repair agent for patch generation. On 300 real-world SEC-Bench cases, AgenticRepair achieves a 73 percent successful repair rate with sanitizer-based patch verification.


논문 제목: SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response
게시 날짜: 2026-07-29 11:32:23
저자: ['Lehan Wang', 'Boli Chen', 'Ruixue Ding', 'Pengjun Xie', 'Jinwei Huang', 'Zhendong Liu', 'Shuo Wang', 'Tao Lei', 'Xin Ouyang', 'Xiaomeng Li']
URL: https://arxiv.org/pdf/2607.26791
요약: This paper introduces SecRespond, a benchmark for evaluating LLM agents on real-world post-compromise incident-response tasks. Agents analyze forensic disk snapshots, alerts, vulnerability scans, and system baselines to identify intrusions and generate remediation plans. The authors evaluate 23 frontier LLMs across 10 compromised cloud-host environments and find that current agents struggle with silent intrusions and verified remediation.


논문 제목: ALIBI: Adaptive Agentic Attacks on LLM-Based Vulnerability Detectors via Adversarial Code Comments
게시 날짜: 2026-07-27 18:13:28
저자: ['Zixuan Wu', 'Cristina Nita-Rotaru']
URL: https://arxiv.org/pdf/2607.24964
요약: This paper studies attacks against LLM-based vulnerability detectors. It introduces ALIBI, an adaptive black-box attack framework in which a coding agent inserts vulnerabilities and adversarial source-code comments designed to manipulate the detector reasoning. Across 125 real-world vulnerabilities, attack success rates exceed 90 percent for all evaluated detectors. Architectural isolation and comment sanitization are more effective than prompt-level defenses.


논문 제목: Just Testing, Move Along: Evasion of LLM-based System Log Interpretation by Prompt Injection
게시 날짜: 2026-07-27 08:59:00
저자: ['Max Landauer', 'Florian Skopik', 'Markus Wurzenberger', 'Franciszek Górski', 'Mateusz Krzysztoń']
URL: https://arxiv.org/pdf/2607.24174
요약: This paper evaluates prompt-injection attacks against LLM-based system-log analysis in Security Operations Center workflows. Attackers insert malicious instructions into log entries so that LLMs interpret genuine indicators of compromise as benign activity. Experiments with multiple state-of-the-art LLMs show that optimized log injections can successfully evade detection. The generated explanations may nevertheless provide signals for identifying the manipulation.

```

### 실습: 요약 프롬프트와 체인 구성

```python
# 요약 프롬프트와 체인 만들기
summary_prompt = ChatPromptTemplate(
    [
        ('system', '''보안 분야의 LLM 관련 논문 목록이 주어집니다.
AI 트렌드 리포트 형식의 뉴스레터를 작성하세요.'''),
        ('human','''{context}''')
    ]
)
summary_chain = summary_prompt | gpt5 | parser
```

```python
# LLM 페이퍼 요약 출력하기
newsletter = summary_chain.invoke(context)
print(newsletter)
```

**실행 결과** (뉴스레터 전문):

# 🔐 AI Security Trend Report
### 에이전틱 보안의 확장과 새로운 공격 표면
**리서치 브리핑 | 2026년 7월 27일–31일**

## 한눈에 보는 핵심

이번 주 논문들은 LLM 보안 에이전트가 **취약점 탐지 이후의 패치 생성과 침해사고 대응까지 역할을 확대**하고 있음을 보여줍니다. 다만 에이전트가 처리하는 코드 주석과 시스템 로그가 새로운 프롬프트 인젝션 경로로 악용되면서, 자동화 수준이 높아질수록 **입력 데이터의 신뢰 경계와 결과 검증**이 더 중요해지고 있습니다.

- **취약점 자동 수정:** 여러 전문 에이전트가 코드 구조·실행 정보·커밋 이력을 수집한 결과, 실제 취약점의 73%를 성공적으로 수정했습니다.
- **침해사고 대응:** 최신 모델들도 조용한 침입을 찾아내고 실제로 검증된 복구 조치를 제시하는 데 어려움을 보였습니다.
- **코드 주석 기반 공격:** 적대적 주석을 이용한 공격이 평가된 모든 취약점 탐지기에서 90% 이상의 성공률을 기록했습니다.
- **로그 프롬프트 인젝션:** 공격자가 로그에 명령을 삽입해 실제 침해 지표를 정상 행위로 오인하게 만들 수 있었습니다.
- **방어의 중심 이동:** 단순한 시스템 프롬프트 강화보다 입력 정제, 권한 격리, 독립 검증과 같은 아키텍처 수준의 통제가 더 효과적인 방향으로 제시됩니다.

---

## 1. AgenticRepair: 전문 에이전트 협업으로 취약점 패치 자동화

**AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair**
Michael Fu, Qiyue Mei, Patanamon Thongtanunam, Kla Tantithamthavorn
2026년 7월 31일 · [논문 보기](https://arxiv.org/pdf/2607.29422)

AgenticRepair는 취약점 수정에 필요한 프로그램 맥락을 하나의 모델이 모두 처리하는 대신, 세 종류의 전문 LLM 하위 에이전트가 나누어 수집하는 멀티에이전트 프레임워크입니다.

### 작동 방식
- **코드 구조 에이전트:** 관련 함수, 호출 관계 등 정적 구조 파악
- **실행 맥락 에이전트:** 런타임 동작과 오류 재현 정보 수집
- **커밋 이력 에이전트:** 과거 변경 내역과 개발 의도 추적
- **수정 에이전트:** 통합된 맥락을 바탕으로 패치 생성
- **검증 단계:** sanitizer 기반으로 패치 유효성 확인

300개의 실제 SEC-Bench 사례에서 **73%의 성공적인 수정률**을 기록했습니다.

### 왜 중요한가
LLM 기반 코드 수정의 병목은 단순히 코드를 생성하는 능력이 아니라, 어떤 정보가 수정에 필요한지 찾아내는 데 있습니다. 다만 sanitizer를 통과한 패치가 기능적 회귀, 새로운 취약점 또는 장기적인 유지보수 문제까지 해결했다는 의미는 아니며, 실제 도입에는 다층 검증이 필요합니다.

---

## 2. SecRespond: 침해 이후 대응 능력을 평가하는 현실적 벤치마크

**SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response**
Lehan Wang 외 9인 · 2026년 7월 29일 · [논문 보기](https://arxiv.org/pdf/2607.26791)

연구진은 **10개의 침해된 클라우드 호스트 환경에서 23개 프런티어 LLM**을 평가했습니다. 그 결과, 현재 에이전트는 눈에 띄는 경보가 적은 **조용한 침입(silent intrusion)**을 찾아내는 데 취약했고, 제안한 복구 조치가 실제로 문제를 해결했는지 검증하는 능력도 제한적이었습니다.

### 핵심 시사점
SOC에서 LLM은 당분간 완전 자율 대응 주체보다 증거 정리·조사 가설 제안·복구 계획 초안·분석가 의사결정 지원 역할에 더 적합합니다. 차단·계정 폐기·시스템 격리와 같은 고위험 조치는 사람의 승인과 사후 검증을 유지해야 합니다.

---

## 3. ALIBI: 코드 주석으로 취약점 탐지기의 추론을 교란

**ALIBI: Adaptive Agentic Attacks on LLM-Based Vulnerability Detectors via Adversarial Code Comments**
Zixuan Wu, Cristina Nita-Rotaru · 2026년 7월 27일 · [논문 보기](https://arxiv.org/pdf/2607.24964)

125개의 실제 취약점을 대상으로 한 실험에서 평가된 모든 탐지기에 대해 **90%가 넘는 공격 성공률**이 보고됐습니다. LLM은 코드와 자연어 주석을 함께 해석하는데, 주석이 모델 행동을 유도하는 명령으로 작동하는 **간접 프롬프트 인젝션** 문제입니다.

### 방어 결과
코드/주석 분석 경로 분리, 주석 제거·정제, 신뢰되지 않은 콘텐츠의 권한 제한, 분석 단계 간 격리 같은 **아키텍처 수준의 대책**이 프롬프트 기반 방어보다 효과적이었습니다.

---

## 4. "Just Testing, Move Along": 시스템 로그도 프롬프트 인젝션 통로가 된다

**Just Testing, Move Along: Evasion of LLM-based System Log Interpretation by Prompt Injection**
Max Landauer 외 4인 · 2026년 7월 27일 · [논문 보기](https://arxiv.org/pdf/2607.24174)

공격자는 로그 항목 안에 악성 지시를 삽입해, 실제 침해 지표를 정상적인 테스트나 무해한 활동으로 설명하도록 모델을 유도합니다. 최적화된 로그 인젝션이 탐지를 성공적으로 회피할 수 있었지만, 모델이 생성한 설명 자체에는 조작을 탐지할 단서(과도한 정상화, 불필요한 자기합리화 등)가 남을 수 있었습니다. 다만 공격자가 제어할 수 있는 로그 필드는 기본적으로 **신뢰할 수 없는 데이터**로 취급해야 합니다.

---

# 공통 트렌드 분석

**① 보안 LLM의 초점이 '탐지'에서 '행동'으로 이동** — AgenticRepair와 SecRespond는 LLM이 취약점 분류를 넘어 패치 생성, 침해 조사, 복구 계획 수립으로 진입하고 있음을 보여줍니다. **정확한 답변**보다 **검증 가능한 행동**이 핵심 평가 기준이 되어야 합니다.

**② 더 많은 컨텍스트가 항상 더 안전한 것은 아니다** — 코드 주석과 로그처럼 공격자가 조작할 수 있는 컨텍스트는 모델을 속이는 경로가 됩니다. 컨텍스트 엔지니어링은 "누가 이 데이터를 생성했는가?", "공격자가 내용을 통제할 수 있는가?"를 중심으로 설계해야 합니다.

**③ 프롬프트 방어만으로는 부족** — 보다 강력한 방어는 모델 바깥(입력 출처별 신뢰 수준 지정, 구조적 분리, 도구 호출 권한 최소화, 독립 검증, 사람의 승인)에서 이뤄져야 합니다.

## Bottom Line

**LLM 보안 에이전트의 성능은 풍부한 맥락과 전문화된 역할 분담으로 향상되지만, 그 맥락 자체가 공격자가 조작할 수 있는 새로운 입력면이 됩니다.** 앞으로의 경쟁력은 (1) 신뢰도와 출처를 반영한 컨텍스트 설계, (2) 최소 권한 기반의 에이전트 아키텍처, (3) 실행 결과를 독립적으로 확인하는 검증 체계에서 결정될 가능성이 큽니다.

---

> 💡 **전체 파이프라인 요약**: `get_arxiv_papers()`(검색) → `classify_chain.batch()`(분류) → `for` 루프 필터링(활용 가능한 데이터로 정제) → `summary_chain.invoke()`(요약/생성)로 이어지는 4단계가, README Lab #2의 "분류 프롬프트 구성 → 분류 체인 연결 → (평가 후) 개선"을 하나의 완결된 실전 워크플로로 만든 것입니다. 흥미롭게도 이 뉴스레터의 "공통 트렌드 분석"(신뢰할 수 없는 컨텍스트, 프롬프트 인젝션, 최소 권한)은 Lab 6의 Agentic RAG, Lab 7의 SQL 실행 안전장치(Human-in-the-Loop)에서 실제로 다시 다루는 주제이기도 합니다.

---

## Part 7. 정리

### 7.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | `init_chat_model()`로 2개 LLM 비교 | `gpt-4.1-mini`와 `gpt-5.2`가 "오늘 날짜"에 서로 다르게(둘 다 부정확하게) 답변 — LLM의 시간 정보 한계 실증 |
| 2 | `Prompt \| LLM` 체인 | 불필요한 입력 변수는 자동 무시, `.invoke().messages`로 실제 프롬프트 확인 가능 |
| 3 | [실습] 2-변수 체인 (2종) | 번역 병기 생성, 캐릭터 대화 생성 — Prefix Caching을 고려한 변수 배치 팁 확인 |
| 4 | `Prompt \| LLM \| Parser` | `StrOutputParser` 추가로 `AIMessage` 대신 순수 문자열 반환 |
| 5 | Arxiv 실시간 검색 | `'Security'` 검색 → 5편(그중 4편만 실제 LLM 관련) — 키워드 검색의 한계 실증 |
| 6 | 분류 체인(`classify_chain.batch`) | 5편을 "LLM"/"Not LLM"으로 분류, 결과 4:1 |
| 7 | 분류 결과 필터링 → 요약 | 4편만 골라 AI 트렌드 뉴스레터 생성 — 분류를 후속 작업의 전처리로 활용 |

### 7.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 확인한 것 |
|---|---|
| LangChain + LLM Provider(p.10), `init_chat_model()`(p.13) | `gpt-4.1-mini`/`gpt-5.2` 두 모델을 동일 인터페이스로 교체·비교 |
| LLM 모델의 한계(p.21) | "오늘 날짜"에 대한 두 모델의 부정확한 답변으로 Hallucination을 실제로 확인 |
| LangChain Document Loader — API 로더(p.29) | `arxiv` 패키지로 실시간 논문 검색(외부 API 연동) |
| Searching: 키워드 기반(Lexical) 검색의 한계(p.32) | `'Security'` 키워드 검색이 LLM 무관 논문("CWEEP")까지 포함 → 후처리 분류 필요성 |
| Lab #2 "분류 프롬프트 구성/분류 체인 연결/평가와 개선"(p.18) | `classify_prompt`+`classify_chain.batch()` 분류 → 필터링 → `summary_chain`으로 활용까지 전체 구현 |

### 7.3 참고 자료

- `arxiv` 파이썬 패키지: <https://pypi.org/project/arxiv/>
- LangChain Prompt Templates: <https://python.langchain.com/docs/concepts/prompt_templates/>
- LLM Provider의 Prompt Caching(참고: OpenAI): <https://platform.openai.com/docs/guides/prompt-caching>
- 원본 실습 노트북: `[강의내용]_2_LangChain을_이용한_데이터_분류와_전처리.ipynb`

### 7.4 다음 단계

- **Lab 1** (README p.14, "LangChain 기본 구조")은 이번 실습에서 사용한 `ChatPromptTemplate`, `batch()`, `stream()` 등 LCEL 기초를 더 앞선 단계에서 다룹니다 — 순서상 이번 Lab 2보다 먼저 학습하는 것이 자연스럽습니다.
- **Lab 3** (`[강의내용]_3_LangChain을_이용한_데이터_생성과_처리.ipynb`)에서는 이번 Part 5에서 지적한 "문자열 포함 여부로 분류 결과를 파싱하는 취약한 방식"의 해결책인 **Pydantic/`with_structured_output()`**을 자세히 다룹니다 — 분류 결과를 `Literal['LLM', 'Not LLM']` 같은 고정 필드로 강제하면 이번 실습의 `'분류 결과: LLM' in classification_result[i]` 같은 문자열 매칭보다 훨씬 안전합니다.
- 실무 팁: 프로덕션 분류 파이프라인에서는 (1) 분류 레이블을 자유 텍스트가 아닌 **Enum/Literal 스키마로 고정**하고, (2) 소량의 정답 데이터로 분류 정확도를 **주기적으로 검증**하며, (3) 이번 실습처럼 **키워드 기반 1차 검색 + LLM 기반 2차 분류**의 2단계 필터링을 결합하는 것이 비용과 정확도의 균형을 맞추는 실전적인 방법입니다.
