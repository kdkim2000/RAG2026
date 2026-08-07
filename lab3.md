# Lab 3. LangChain을 이용한 데이터 생성과 처리

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"LangChain의 Structured Output"(p.16)**, **"특수한 Runnables"(p.17)** 이론을 `[강의내용]_3_LangChain을_이용한_데이터_생성과_처리.ipynb`의 실제 실행 결과와 결합하여 재구성한 것입니다. 노트북 파일명은 `[강의내용]_3_...`이지만, 노트북 첫 셀의 제목이 정확히 **"# [실습] LangChain을 이용한 데이터 생성과 처리"**로 되어 있어, README 부록 "실습(Lab) 목록"의 **Lab #3 "LangChain을 이용한 데이터 생성과 처리"(p.19, Day1-2)**에 해당함을 확인할 수 있습니다.
>
> - README Lab #3 소개: "LLM의 구조화된 출력을 사용하여, 다양한 데이터 전처리하기 / 적은 수의 샘플을 사용해, 유사한 데이터 생성하기"

## 학습 목표

- JSON/Pydantic 기반 **구조화된 출력**으로 LLM 응답을 별도 후처리 없이 곧바로 프로그램에서 다룰 수 있는 형태로 받는 방법을 익힌다.
- 스키마 없는 JSON 파서의 한계(**실행마다 출력 형식이 달라짐**)를 실제로 확인하고, Pydantic으로 스키마를 고정해야 하는 이유를 체감한다.
- `with_structured_output()`과 `PydanticOutputParser` 두 가지 구조화된 출력 방식의 차이를 비교한다.
- **"개요 생성(outliner) → 섹션별 병렬 작성(writer.batch) → 합치기"**라는, 구조화된 출력을 다음 체인의 입력으로 연결하는 실전 패턴을 구현한다 — README p.19의 "적은 수의 샘플을 사용해 유사한 데이터 생성하기"를 실제로 구현한 사례이다.
- `RunnablePassthrough` / `RunnableParallel` / `.assign()` 7가지 실습으로 "특수한 Runnables"(p.17) 이론을 완전히 체화한다.

## Part 0. 개요와 노트북 구조

**노트북 52개 셀의 흐름**:
1. 환경설정, LLM 준비(`init_chat_model`)
2. **JsonOutputParser**로 스키마 없는 JSON 출력 만들기 → 실행마다 형식이 달라지는 문제 확인
3. **Pydantic**으로 스키마를 고정한 JSON 출력 만들기 (`JsonOutputParser(pydantic_object=...)`)
4. **`with_structured_output()`**으로 파서 없이 구조화된 출력 만들기, `PydanticOutputParser`와 비교
5. **[실습] 보고서 개요 생성 후 섹션별 글 작성하기** — 구조화된 출력을 스캐폴딩으로 활용한 긴 글 자동 생성
6. **Runnables 심화** — `RunnablePassthrough`, `RunnableParallel`, `.assign()` 7가지 실습

---

## Part 1. 환경 준비

### 이론: Runnable과 Invoke (p.13)

> - `init_chat_model()`을 통한 API 통합 연결 — `model_provider`, `model_name` 전달

### 실습: 패키지 설치와 LLM 준비

```python
%pip install langchain langchain_openai dotenv rich -q
```

**실행 결과**: `Note: you may need to restart the kernel to use updated packages.`

```python
import os
from dotenv import load_dotenv
load_dotenv(override=True)

if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')
```

**실행 결과**: `OpenAI API 키 확인`

```python
from langchain.chat_models import init_chat_model

gpt_llm = init_chat_model(
    "gpt-5.2", reasoning_effort='low')
```

> 💡 README p.13이 설명하는 `init_chat_model()`을 그대로 사용합니다 — `ChatOpenAI(...)`를 직접 생성하는 대신, Provider에 상관없이 동일한 함수로 모델을 초기화하는 방식입니다. 이번 실습에서는 `rich` 패키지도 함께 설치했는데, 이는 Part 4에서 구조화된 객체를 예쁘게 출력(pretty-print)하기 위한 것입니다.

---

## Part 2. JsonOutputParser로 JSON 형식의 출력 만들기

### 이론: LangChain의 Structured Output (p.16)

> - 기존의 LLM은 단순 Message 리스트를 전달하는 방식
> - 구조화된 출력은 별도의 후처리 필요 없이 DB/프롬프트에 바로 연결할 수 있음 — LLM 프로그램 구조 고도화에서 필수적
> - 다양한 파서를 통한 출력 형식 변환 — JSON, Pydantic, DateTime, ...

### 실습: JsonOutputParser 기본 사용법

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

jsonparser = JsonOutputParser()
```

> JSON 파서의 역할은 JSON 규격에 맞는 텍스트를 Dict 형식으로 변환하는 것으로, 실제 형식에 대한 조건을 프롬프트로 전달해야 합니다.

```python
jsonparser.get_format_instructions()
```

**실행 결과**: `'Return a JSON object.'`

> ⚠️ **주목할 점**: `JsonOutputParser()`를 스키마 없이 쓰면, LLM에게 전달되는 형식 지시가 **"Return a JSON object."라는 한 문장뿐**입니다. 즉 "JSON으로 응답하라"는 것 외에는 **어떤 필드를 어떤 이름·형식으로 만들어야 하는지 전혀 지정하지 않습니다.** 이 점이 바로 아래에서 실제로 문제가 되는 지점입니다.

### 실습: 레시피 생성 체인 — 첫 번째 실행

```python
recipe_template = ChatPromptTemplate([
    ('system','당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
    ('user','''저는 다음의 재료와 조건을 이용한 환상적인
퓨전 다이닝을 만들고 싶습니다. 1가지 메뉴만 추천해주세요!
레시피에 대한 정보를 JSON 형식으로 출력해주세요.

[재료]: {ingredient}
[조건]: {condition}
''')
])

recipe_chain = recipe_template | gpt_llm | jsonparser
```

```python
response = recipe_chain.invoke({'ingredient':'콜라, 고수, 새우', 'condition':'디저트'})
response
```

**실행 결과**:
```python
{'menu_name': '콜라 카라멜 쉬림프 프랄린 & 고수-라임 그라니타',
 'type': 'dessert',
 'fusion_concept': '콜라를 졸여 만든 카라멜(탄산 향+스파이스 뉘앙스)에 새우를 바삭하게 코팅해 프랄린처럼 만들고, 고수의 허브향을 라임과 함께 얼려 그라니타로 곁들이는 달콤-짭짤-상큼 퓨전 디저트',
 'servings': 2,
 'ingredients': [{'group': '콜라 카라멜 쉬림프',
   'items': [{'name': '새우(껍질 제거, 내장 제거)', 'amount': '10마리(중하 크기)'},
    {'name': '콜라', 'amount': '250ml'},
    {'name': '버터', 'amount': '20g'},
    {'name': '갈색설탕(선택)', 'amount': '1큰술'},
    {'name': '소금', 'amount': '꼬집'},
    {'name': '라임 제스트(선택)', 'amount': '약간'},
    {'name': '전분(감자/옥수수)', 'amount': '2큰술'},
    {'name': '식용유', 'amount': '적당량(팬에 0.5cm)'}]},
  {'group': '고수-라임 그라니타',
   'items': [{'name': '물', 'amount': '200ml'},
    {'name': '설탕', 'amount': '35g'},
    {'name': '라임즙', 'amount': '2큰술'},
    {'name': '고수(잎 위주)', 'amount': '한 줌(약 15g)'},
    {'name': '소금', 'amount': '아주 약간'}]},
  {'group': '마무리(선택)',
   'items': [{'name': '화이트초콜릿(얇게 깎기)', 'amount': '10g'},
    {'name': '볶은 코코넛칩 또는 깨', 'amount': '1큰술'}]}],
 'steps': [{'part': '고수-라임 그라니타',
   'instructions': ['냄비에 물+설탕을 넣고 약불에서 설탕만 녹여 시럽을 만든 뒤 완전히 식힌다.',
    '식힌 시럽에 라임즙, 고수 잎, 소금 아주 약간을 넣고 블렌더로 곱게 간다.',
    '얕은 트레이에 붓고 냉동실에 3~4시간 얼린다. 30~40분마다 포크로 긁어 결정을 만들어 그라니타 질감을 만든다.']},
  {'part': '콜라 카라멜 소스',
   'instructions': ['팬에 콜라를 붓고 중불에서 1/3 이하로 줄 때까지 진득하게 졸인다(거품이 잦아들고 시럽처럼 되면 OK).',
    '불을 약하게 줄이고 버터, 갈색설탕(선택), 소금 꼬집을 넣어 윤기 있게 섞는다. 필요하면 라임 제스트를 아주 조금 넣어 향을 올린다.']},
  {'part': '새우 프랄린(디저트식 코팅)',
   'instructions': ['새우 물기를 최대한 닦고 전분을 얇게 묻힌다(바삭한 코팅 목적).',
    '팬에 식용유를 달군 뒤(중불) 새우를 40~60초씩 짧게 튀기듯 익혀 겉을 바삭하게 만든다. 꺼내서 키친타월에 기름을 뺀다.',
    '바삭한 새우를 카라멜 팬에 넣고 10~15초 빠르게 코팅한다(오래 두면 눅눅해짐). 바로 유산지 위로 옮겨 서로 붙지 않게 식힌다.']}],
 'plating': ['차가운 접시에 그라니타를 먼저 한 스푼(또는 타원형으로) 올린다.',
  '그 옆에 콜라 카라멜 쉬림프를 5마리 정도 '프랄린'처럼 쌓는다.',
  '선택 재료로 화이트초콜릿을 얇게 깎아 눈처럼 뿌리고, 코코넛칩(또는 깨)을 소량 흩뿌린다.',
  '마지막에 그라니타 위에 아주 작은 고수 잎 1~2장을 올려 향의 힌트를 준다.'],
 'time_required': {'prep': '15분',
  'cook': '15분',
  'freeze': '3~4시간(그라니타)',
  'total_active_time': '약 30분'},
 'taste_profile': {'sweetness': '중간~높음(콜라 카라멜)',
  'saltiness': '약간(카라멜의 소금+새우 자체 감칠맛)',
  'acidity': '중간(라임+그라니타)',
  'herbal_note': '뚜렷함(고수)',
  'texture': '바삭(새우) + 서늘한 결정감(그라니타)'},
 'success_tips': ['새우는 '짧게' 익혀야 디저트에서도 질감이 부드럽고 비리지 않다(과열 금지).',
  '카라멜 코팅은 10~15초 내로 끝내고 바로 식혀 바삭함을 유지한다.',
  '고수 향이 강하게 느껴지는 게 의도지만, 부담되면 고수 양을 30% 줄이고 라임즙을 조금 늘린다.'],
 'allergens': ['새우(갑각류)', '유제품(버터, 선택 시)', '초콜릿(선택 시)']}
```

```python
# Dict 구조: 추출 가능
response['menu_name']
```

**실행 결과**: `'콜라 카라멜 쉬림프 프랄린 & 고수-라임 그라니타'`

`JsonOutputParser`가 문자열이 아닌 **Dict**를 반환하므로, `response['menu_name']`처럼 곧바로 특정 필드에 접근할 수 있습니다 — README p.16의 "별도의 후처리 필요 없이 DB/프롬프트에 바로 연결할 수 있음"이 실제로 확인되는 지점입니다.

### ⚠️ 실제로 확인된 문제: 실행마다 형식이 달라짐

> Json으로 파싱하는 방법은 활용도가 높지만, 실행할 때마다 결과뿐만 아니라 형식도 달라진다는 문제가 있습니다.

```python
response = recipe_chain.invoke({'ingredient':'문어, 피넛버터', 'condition':'메인 요리'})
response
```

**실행 결과**:
```python
{'menu_name_ko': '피넛버터 라케로 글레이즈드 문어 스테이크',
 'menu_name_en': 'Peanut-Butter Lacquered Octopus Steak',
 'category': '메인 요리',
 'fusion_concept': {'cuisines': ['한국', '서아프리카(마페 감성)', '일본(테리야키식 글레이즈)'],
  'idea': "피넛버터를 고소한 소스 베이스로 쓰고, 간장·식초·고추로 밸런스를 맞춘 뒤 문어에 '라케(코팅) 글레이즈'를 여러 번 발라 윤기와 풍미를 극대화한 메인 디시"},
 'servings': 2,
 'difficulty': '중',
 'time': {'prep_minutes': 15, 'cook_minutes': 35, 'total_minutes': 50},
 'ingredients': [{'name': '문어(자숙 또는 생문어)', 'amount': '450g', 'notes': '생문어면 부드럽게 삶는 과정 포함'},
  ... (총 13개 재료) ...],
 'optional_garnish_and_sides': [{'name': '오이 리본/채', 'purpose': '산뜻한 식감과 쿨링'}, ...],
 'equipment': ['냄비', '팬(무쇠/스테인리스 권장)', '볼', '집게', '칼', '도마'],
 'recipe_steps': [{'step': 1, 'title': '문어 준비(생문어일 경우 부드럽게 삶기)', 'detail': '...'}, ... (5단계) ...],
 'chef_notes': ['피넛버터는 단독으로는 무겁기 쉬우므로 식초(산)와 간장(짠맛), 고추장(매콤/발효감)으로 밸런스를 잡는 것이 핵심입니다.', ...],
 'flavor_profile': ['고소함', '감칠맛', '은은한 단맛', '산미', '매콤함', '스모키한 시어링 향'],
 'plating_style': '파인 다이닝 스타일(베이스-단백질-글레이즈-상큼한 가니시-크런치 토핑)',
 'allergen_info': ['땅콩', '대두(간장)', '조개류/연체류(문어)'],
 'pairing_suggestion': {'drink': '드라이한 화이트 와인(소비뇽 블랑) 또는 하이볼',
  'reason': '산도와 탄산/드라이함이 피넛 글레이즈의 농도를 깔끔하게 정리'}}
```

> 🐛 **실제로 확인된 스키마 불일치**: 첫 번째 호출과 두 번째 호출을 비교하면, 같은 프롬프트 템플릿(`recipe_chain`)으로 생성했는데도 **최상위 키 구성이 완전히 다릅니다.**
>
> | 필드 | 1차 호출(콜라·고수·새우) | 2차 호출(문어·피넛버터) |
> |---|---|---|
> | 메뉴명 | `menu_name` (문자열 1개) | `menu_name_ko` + `menu_name_en` (2개, 다국어) |
> | 컨셉 | `fusion_concept` (문자열) | `fusion_concept` (`cuisines` 리스트 + `idea` 문자열을 담은 객체) |
> | 재료 | `ingredients` (그룹별로 묶인 2중 구조) | `ingredients` (평평한 1차원 리스트) |
> | 조리법 | `steps` (파트별 묶음) | `recipe_steps` (단계 번호가 있는 평평한 리스트) |
>
> `jsonparser.get_format_instructions()`가 `"Return a JSON object."`라는 최소한의 지시만 주었기 때문에, LLM은 매번 **재료·조건에 맞춰 스스로 판단한 구조**로 JSON을 생성합니다. 이는 사람이 읽기에는 오히려 자연스럽고 풍부하지만, **이 응답을 코드에서 `response['menu_name']`처럼 고정된 키로 파싱해야 하는 프로그램 입장에서는 매번 `KeyError`가 날 수 있는 심각한 문제**입니다. → 이 문제를 해결하는 방법이 바로 다음 Part의 **Pydantic 스키마 고정**입니다.

---

## Part 3. Pydantic을 이용해 출력 형식 지정하기

> pydantic은 데이터 형식에 제약조건을 두고 이를 준수하는지 검증하는 라이브러리입니다.

### 실습: Pydantic 모델로 스키마 정의

```python
from pydantic import BaseModel, Field
# pydantic 연동

class Recipe(BaseModel):
    name: str = Field(description="음식 이름")
    # name: 문자열, 설명은 "음식 이름"
    difficulty: str = Field(description="만들기의 난이도")

    origin: str = Field(description="원산지")
    ingredients: list[str] = Field(description="재료")
    # ingredients: 문자열 리스트, 설명은 "재료"

    instructions: list[str] = Field(description="조리법")
    tip: str = Field(description='실패하는 5가지 시나리오')
```

```python
parser = JsonOutputParser(pydantic_object=Recipe)
```

```python
print(parser.get_format_instructions())
```

**실행 결과**:
```
STRICT OUTPUT FORMAT:
- Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
- Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
- Do not prepend or append any text (e.g., do not write "Here is the JSON:").
- The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema (shown in a code block for readability only — do not include any backticks or Markdown in your output):
```
{"properties": {"name": {"description": "음식 이름", "title": "Name", "type": "string"}, "difficulty": {"description": "만들기의 난이도", "title": "Difficulty", "type": "string"}, "origin": {"description": "원산지", "title": "Origin", "type": "string"}, "ingredients": {"description": "재료", "items": {"type": "string"}, "title": "Ingredients", "type": "array"}, "instructions": {"description": "조리법", "items": {"type": "string"}, "title": "Instructions", "type": "array"}, "tip": {"description": "실패하는 5가지 시나리오", "title": "Tip", "type": "string"}}, "required": ["name", "difficulty", "origin", "ingredients", "instructions", "tip"]}
```
```

> 💡 **Part 2와의 극명한 대비**: Part 2에서는 형식 지시가 `"Return a JSON object."` 한 문장이었지만, `pydantic_object=Recipe`를 지정하는 순간 지시문이 **전체 JSON Schema(6개 필드의 이름·타입·설명 + "필드 순서를 어기지 마라", "코드펜스를 쓰지 마라" 같은 세부 규칙)**까지 포함하는 수십 줄짜리 지시문으로 바뀝니다. 이 지시문 전체가 매 호출 시 LLM에게 그대로 전달되어, **비로소 스키마가 고정**됩니다.

### 실습: 프롬프트에 형식 지시 포함하기

```python
recipe_template2 = ChatPromptTemplate([
    ('system','당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
    ('user','''저는 다음의 재료를 이용한 실험적인 음식을 만들고 싶습니다. 추천해주세요!
    레시피에 대한 정보를 JSON 형식으로 출력해주세요. 결과는 한국어로 작성하세요.
재료: {ingredient}
출력 형식 조건: {instruction}''')
])

recipe_chain2 = recipe_template2 | gpt_llm | parser
```

```python
recipe_chain2.invoke({'ingredient':'생강', 'instruction':parser.get_format_instructions()})
```

**실행 결과**:
```python
{'name': '생강 코지 버터 미소 카라멜 팝콘',
 'difficulty': '중',
 'origin': '일본 코지·미소 발효 풍미 + 서구식 카라멜 팝콘 퓨전',
 'ingredients': ['팝콘용 옥수수 100g', '식용유 2큰술', '무염버터 60g', '설탕 120g', '물엿(또는 꿀) 30g',
  '미소(된장) 1큰술', '생강(생강즙 또는 강판) 1~2큰술', '간장 1작은술', '베이킹소다 1/4작은술',
  '코지(건조 쌀코지) 가루 1~2큰술(선택)', '소금 한 꼬집', '검은깨 또는 김가루 약간(선택)'],
 'instructions': ['큰 냄비에 식용유를 두르고 옥수수를 넣은 뒤 뚜껑을 덮어 팝콘을 튀긴다. 다 튀겨지면 넓은 볼이나 오븐팬으로 옮긴다.',
  '두꺼운 바닥 냄비에 버터를 녹인 뒤 설탕과 물엿을 넣고 중불에서 저어가며 끓인다.',
  '시럽이 끓어오르면 불을 중약불로 낮추고 미소, 간장, 생강(즙/강판)을 넣어 완전히 풀어준다.',
  '거품이 잦아들고 색이 연한 호박색이 되면 불을 끄고 베이킹소다와 소금 한 꼬집을 넣어 빠르게 섞는다.',
  '즉시 팝콘 위에 소스를 고루 부어 주걱으로 빠르게 버무린다.',
  '오븐 120~130°C에서 20~25분 정도 말리듯이 구워 바삭하게 만든다.',
  '완전히 식힌 뒤 검은깨나 김가루를 소량 뿌려 마무리한다. 밀폐 용기에 보관한다.'],
 'tip': '실패하는 5가지 시나리오: 1) 카라멜이 타서 쓴맛이 남음. 2) 생강을 너무 많이 넣어 매운 비누맛/섬유감이 도드라짐. 3) 미소를 고온에서 오래 끓여 짠맛만 남음. 4) 베이킹소다를 미리 넣거나 과다 사용해 금속성/비누향. 5) 코팅 후 건조가 부족해 눅눅해짐.'}
```

**이제 필드명이 정확히 `name`/`difficulty`/`origin`/`ingredients`/`instructions`/`tip` 6개로 고정**되었습니다 — Part 2에서 실행마다 달라졌던 스키마가, Pydantic 모델을 지정하는 순간부터는 재료가 바뀌어도(콜라·새우 → 생강) **항상 동일한 구조**로 나옵니다.

### 실습: `.partial()`로 인자 일부를 미리 채우기

> partial을 통해 먼저 일부를 입력할 수도 있습니다.

```python
recipe_template2 = ChatPromptTemplate([
    ('system','당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
    ('user','''저는 {ingredient}를 이용한 실험적인 음식을 만들고 싶습니다. 추천해주세요!
    레시피에 대한 정보를 JSON 형식으로 출력해주세요. 결과는 한국어로 작성하세요.
     {instruction}''')
]).partial(instruction=parser.get_format_instructions())

recipe_chain2 = recipe_template2 | gpt_llm | parser

recipe_chain2.invoke('감자')
# partial은 기본값: instruction을 다시 덮어쓸 수도 있음
```

**실행 결과** (일부):
```python
{'name': '감자 코지(麹) 카라멜 미소 브륄레 + 김오일',
 'difficulty': '중상',
 'origin': '퓨전(일본 발효 × 프랑스 디저트 × 한국 김)',
 'ingredients': ['감자 500g(전분 많은 품종 추천)', '우유 250ml', '생크림 200ml', '달걀노른자 4개', ...],
 'instructions': ['감자 준비: 감자를 껍질째 찐 뒤 껍질을 벗기고 뜨거울 때 으깬다.', ...],
 'tip': '실패 시나리오 5가지: 1) 코지 보온 온도가 65℃ 이상으로 올라가면 효소가 죽어 당화가 약해진다. ...'}
```

> 💡 `.partial(instruction=...)`으로 프롬프트 변수 하나를 **미리 고정**하면, 이후 체인을 호출할 때는 `{'ingredient': ..., 'instruction': ...}` 딕셔너리 대신 **`'감자'`라는 문자열만** 넘겨도 됩니다 — 남은 변수(`ingredient`)가 하나뿐이면 문자열 하나만으로 자동 매핑됩니다. `instruction`처럼 "항상 같은 값이 들어가는 변수"를 매번 명시적으로 넘기지 않아도 되므로, 체인의 재사용성이 크게 좋아집니다.

---

## Part 4. `with_structured_output()`으로 파서 없이 구조화된 출력 만들기

> 파서를 사용하지 않고, 구조화된 출력을 생성합니다.

### 실습: `with_structured_output(Recipe)`

```python
from rich import print as rprint
structured_llm = gpt_llm.with_structured_output(Recipe)

rprint(structured_llm)
```

**실행 결과** (`rich.print`의 구조 출력, 색상 코드 제거 후 핵심 내용만 정리):
```
RunnableSequence(
    first=_ChatModelBinding(
        bound=ChatOpenAI(
            profile={
                'name': 'GPT-5.2', 'release_date': '2025-12-11',
                'max_input_tokens': 272000, 'max_output_tokens': 128000,
                'structured_output': True, 'tool_calling': True, ...
            },
            model_name='gpt-5.2', reasoning_effort='low', ...
        ),
        kwargs={
            'response_format': <class '__main__.Recipe'>,
            'ls_structured_output_format': {
                'kwargs': {'method': 'json_schema', 'strict': None},
                'schema': {'type': 'function', 'function': {'name': 'Recipe',
                    'parameters': {'properties': {...6개 필드...},
                                   'required': ['name','difficulty','origin','ingredients','instructions','tip'],
                                   'type': 'object'}}}
            }
        },
    ),
    last=RunnableBinding(bound=RunnableLambda(...), custom_output_type=<class '__main__.Recipe'>)
)
```

> 💡 **코드 동작 설명**: `with_structured_output(Recipe)`는 Part 3처럼 프롬프트에 "이런 형식으로 답하라"는 **텍스트 지시문을 추가하는 방식이 아니라**, OpenAI API의 `response_format`(`method: 'json_schema'`) 파라미터에 **Recipe의 JSON Schema를 직접 바인딩**합니다. 즉 모델이 스키마를 지키도록 프롬프트로 "부탁"하는 것이 아니라, **API 차원에서 스키마를 강제**하는 방식입니다 — README p.16의 `with_structured_output()` 활용이 내부적으로 이렇게 동작함을 실제 객체 구조로 확인할 수 있습니다.

```python
recipe_template3 = ChatPromptTemplate([
    ('system','당신은 한국 전통의 재료가 가진 다양한 맛과 특성을 활용합니다.'),
    ('user','''저는 {ingredient}를 이용한 실험적인 음식을 만들고 싶습니다. 추천해주세요!''')
])

recipe_chain3 = recipe_template3 | structured_llm
response = recipe_chain3.invoke("생강")
rprint(response)
```

> ⚠️ **실제로 발견된 이슈 — 출력 인코딩 손상**: 노트북에 기록된 이 셀의 실행 결과(`rich.print`가 출력한 텍스트)를 그대로 확인하면, 한글 부분이 아래와 같이 **손상된 문자(mojibake)로 기록**되어 있습니다.
> ```
> Recipe(
>     name='���� ��ȿ��[...]�۷����� �κ� ������ũ & ���ڻ��� ���ġ ���',
>     difficulty='��',
>     ...
> )
> ```
> **원인 추정**: `rich` 라이브러리가 ANSI 컬러 코드와 함께 텍스트를 콘솔에 출력할 때, 이를 캡처/저장하는 과정에서 **Windows 콘솔의 기본 코드페이지(cp949)와 실제 UTF-8 한글 인코딩이 불일치**하여 발생한 것으로 보입니다 — 이 저장소의 다른 실습(Lab 5~8)에서도 여러 차례 확인된 "Windows 환경 + 한글 텍스트 + 특정 출력 파이프"의 조합에서 반복적으로 나타나는 인코딩 문제의 또 다른 사례입니다.
>
> **실습에서 확인할 수 있는 사실**: 인코딩은 손상되었지만 **구조 자체는 살아있습니다** — `Recipe(name=..., difficulty=..., origin=..., ingredients=[...], instructions=[...], tip=...)` 형태의 **Pydantic 객체**가 정상적으로 생성되었고, 6개 필드가 모두 채워져 있습니다. 이는 Part 3의 `parser.parse()`가 반환하는 **Dict**와 달리, `with_structured_output()`은 **Pydantic 모델 인스턴스 자체**를 반환한다는 차이를 보여줍니다(`response.name`처럼 속성으로 접근 가능). 참고로 (같은 방식으로 생성된) Part 3의 생강 레시피 결과("생강 코지 버터 미소 카라멜 팝콘")로 유추하면, 이 손상된 출력도 유사하게 생강을 활용한 발효·카라멜 계열의 퓨전 메뉴였을 것으로 짐작되지만, **실제 내용은 인코딩 손상으로 복구할 수 없으므로 추측을 사실처럼 단정하지 않습니다.**

### 실습: `PydanticOutputParser` — `with_structured_output`을 지원하지 않는 모델의 대안

> 해당 출력은 Pydantic 클래스 형식으로 생성됩니다. `with_structured_output` 기능을 지원하지 않는 경우, `PydanticOutputParser`를 사용해야 합니다.

```python
from langchain_core.output_parsers import PydanticOutputParser

pydantic_parser = PydanticOutputParser(pydantic_object = Recipe)

recipe_template4 =ChatPromptTemplate([
    ('system','당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
    ('user','''저는 {ingredient}를 이용한 실험적인 음식을 만들고 싶습니다. 추천해주세요!
    레시피에 대한 정보를 JSON 형식으로 출력해주세요. 결과는 한국어로 작성하세요.
     {instruction}''')
])

structured_llm2 = recipe_template4.partial(instruction = pydantic_parser.get_format_instructions()) | gpt_llm | pydantic_parser

response = structured_llm2.invoke('커피')
response
```

**실행 결과**:
```python
Recipe(name='에스프레소-미소 카라멜 된장글레이즈 연어 타코 (커피+일식+멕시칸 퓨전)',
       difficulty='중상 (불 조절과 글레이즈 농도 조절이 핵심)',
       origin='퓨전: 일본(미소/된장) + 멕시코(타코) + 이탈리아(에스프레소) + 북유럽(연어)',
       ingredients=['연어 필렛 300g (껍질 있는 것 추천)', '소금 3g', '후추 약간', '올리브오일 1큰술',
                    '또띠아(옥수수 또는 밀) 6장', '양배추 채 1컵', '라임 1개', '고수(선택) 한 줌',
                    '에스프레소 60ml (또는 진하게 내린 커피 80ml)', '된장(미소) 2큰술', '흑설탕 또는 황설탕 2큰술',
                    '버터 1큰술', '간장 1큰술', '식초(사과식초/쌀식초) 1작은술', '고춧가루 또는 칠리플레이크 1/2작은술',
                    '마요네즈 3큰술', '플레인 요거트 2큰술', '커피가루(곱게 분쇄) 1/2작은술 (소스용, 선택)', '마늘 1쪽(다짐) 또는 마늘가루 1/3작은술'],
       instructions=['연어에 소금·후추로 밑간하고 10분 두어 표면 수분을 정리한다.',
                     '글레이즈 만들기: 작은 팬에 에스프레소, 된장, 설탕, 간장, 식초, 마늘, 칠리를 넣고 중약불에서 저어가며 끓인다.',
                     '거품이 잦아들고 점도가 시럽처럼 걸쭉해지면 불을 끄고 버터를 넣어 윤기를 낸다.',
                     '소스(크레마) 만들기: 마요네즈+요거트를 섞고, 라임즙과 라임 제스트를 넣는다.',
                     '팬을 강불로 예열 후 올리브오일을 두른다. 연어는 껍질면부터 3~4분 굽고, 뒤집어 1~2분만 익힌다.',
                     '불을 중약불로 낮추고 글레이즈를 연어 위에 2~3번 얇게 덧발라가며 30~60초만 빠르게 코팅한다.',
                     '연어를 꺼내 2분 휴지 후 먹기 좋은 크기로 찢거나 썬다.',
                     '또띠아를 마른 팬에 10~20초씩 데워 유연하게 만든다.',
                     '또띠아에 양배추 채를 깔고, 연어를 올린 다음 글레이즈를 한 번 더 가볍게 뿌린다.',
                     '라임 크레마를 올리고 고수(선택)로 마무리한다.'],
       tip='실패하는 5가지 시나리오: (1) 글레이즈를 강불에서 오래 끓여 설탕·커피가 탄 맛이 나고 쓴맛만 남는다. (2) 된장을 많이 넣거나 충분히 풀지 않아 짠 덩어리가 남는다. (3) 연어를 글레이즈와 함께 오래 졸여 과익·비린내가 올라온다. (4) 커피가루를 많이 넣어 텁텁하고 모래 같은 식감이 생긴다. (5) 산미(라임/식초)가 부족해 단짠+커피 쓴맛이 무겁게 느껴진다.')
```

> 💡 이 셀은 표준 Python `repr()` 출력(색상 코드 없이 `print()` 그대로)이라 인코딩 손상 없이 정상 기록되었습니다 — 바로 위 `rich.print` 셀과 대조하면, **문제의 원인이 `rich`의 콘솔 렌더링 경로**라는 추정을 뒷받침합니다. `with_structured_output()`(API 차원 강제)과 `PydanticOutputParser`(프롬프트 지시 + 파싱)는 **둘 다 최종적으로 동일한 `Recipe` Pydantic 객체를 반환**한다는 점도 이 결과로 확인됩니다 — 모델이 `with_structured_output`을 지원하지 않을 때(예: 일부 오픈소스/구형 모델) `PydanticOutputParser`가 대체 수단이 되는 이유입니다.

---

## Part 5. [실습] LLM으로 보고서 개요 생성 후 섹션별 글 작성하기

> Structured Output 구조를 활용해, LLM이 주제에 대한 구획을 먼저 구성하고 해당 구획을 반복문이나 Batch로 각각 입력하여 긴 글을 쓰도록 만들어 보세요.
>
> 1. `with_structured_output`을 통해 주제에 대한 구획 작성하는 체인 `outliner` 만들기
> 2. 섹션별 글 작성 체인 `writer` 만들기
> 3. 반복문이나 `batch()`를 통해 `outliner`의 결과물을 `writer`에 전달하기
> 4. 최종 결과물 합치기

이 실습은 README p.19의 **"적은 수의 샘플을 사용해 유사한 데이터 생성하기"**를 그대로 구현한 것입니다 — "적은 수의 샘플"은 여기서 `outliner`가 만든 **섹션 개요 5개**에 해당하고, 이 5개의 짧은 개요를 씨앗(seed)으로 삼아 **형식이 유사한(같은 `writer_prompt`를 공유하는) 5개의 긴 글을 병렬 생성**합니다.

### 실습: 개요를 위한 Pydantic 스키마와 `outliner`

```python
class Sections(BaseModel):
    topic: str = Field(description="글쓰기 주제")
    sections: list[str] = Field(description="주제에 대한 세부 섹션 개요 리스트 (최대 5개 섹션)")
```

```python
outliner = gpt_llm.with_structured_output(Sections)
outline = outliner.invoke("""
멀티모달 LLM의 발전 과정에 대한 보고서 개요와 목차를 써줘.
각각의 개요는 병렬적 작성이 가능하도록 독립적인 내용을 담아야 하고.
개요만 보고도 내용이 구체적으로 드러나야 해.
목차는 리스트로 작성해.
""")
outline
```

**실행 결과**:
```python
Sections(
  topic='멀티모달 LLM의 발전 과정(연대기·기술축 중심) 보고서 개요 및 목차',
  sections=[
    '정의·범위·평가 관점(멀티모달이 ‘무엇을’ 해결하는가): 텍스트 중심 LLM 대비 입력/출력 모달리티(이미지·오디오·비디오·센서·문서), 과업군(VQA, 캡셔닝, OCR/문서 이해, 시청각 대화, 비디오 내 이벤트 추적), 핵심 난제(정렬·추론·시간축·메모리·도구사용)와 대표 벤치마크(MMBench, SEED, MMMU, MathVista, VQA, TextVQA, DocVQA 등)로 ‘발전’을 측정하는 기준을 먼저 고정',
    '초기 멀티모달 신경망→비전-언어 사전학습(2018~2021): CNN/Transformer 기반 캡셔닝·VQA에서 시작해, 대규모 이미지-텍스트 사전학습(CLIP류의 대조학습, ALIGN 등)과 cross-attention 기반 비전-언어 모델(VilBERT/UNITER류)로 전환되며 ‘공통 임베딩 공간’과 ‘정렬 학습’이 성능·전이의 기반이 된 과정, 데이터(웹 스크랩 이미지-자막)와 학습 목표(contrastive, masked modeling)의 영향 비교',
    'LLM 결합의 1세대(2022~2023): ‘비전 인코더+LLM’ 어댑터 패러다임—고정/미세조정된 비전 인코더 출력(특징 토큰)을 프로젝터(선형/MLP/Q-Former)로 LLM 입력 공간에 매핑해 시각 대화·지시따르기(Instruction Following)를 구현한 흐름(Flamingo, BLIP-2, LLaVA 계열), 정렬 데이터(이미지-설명→대화형 지시 데이터) 제작과 SFT/RLHF 도입이 사용자 경험을 어떻게 바꿨는지, 그리고 환각·세밀 인식 한계가 어디서 발생했는지',
    '고도화 단계(2023~2025): 고해상도·문서·비디오·오디오로의 확장과 ‘에이전트화’—(1) 이미지: 고해상도 타일링/멀티스케일, OCR 결합, 레이아웃 이해; (2) 비디오: 시간적 토큰 압축, 이벤트 세그먼트, 장기 컨텍스트; (3) 오디오: 음성인식/이해/생성의 통합; (4) 도구사용: 브라우징·코드·검색·OCR·ASR을 호출하는 멀티모달 에이전트, 그리고 모델 단독 vs 툴 결합의 성능/비용 트레이드오프',
    '미래 과제와 신뢰성(2025~): 데이터·안전·평가의 재정의—합성데이터/자기학습, 멀티모달 RL, 체인오브쏘트 노출 없이도 검증 가능한 추론(프로그램/툴 기반 검증), 환각·프롬프트 인젝션·저작권/개인정보 이슈, ‘벤치마크 과적합’ 문제와 실제 업무(문서 결재, 고객센터, 의료/제조 시각검사) 중심의 평가 프레임으로의 이동 전망'
  ]
)
```

```python
outline.sections
```

`Sections` 객체는 `outline.sections`로 곧바로 리스트를 꺼낼 수 있습니다(Part 4에서 확인한 Pydantic 객체 접근 방식) — 총 5개의 독립적인 섹션 개요가 생성되었습니다.

### 실습: 섹션 하나를 작성하는 `writer` 체인 (단일 실행)

```python
writer_prompt = ChatPromptTemplate([
    ('human','''보고서 주제에 대해, 하나의 섹션에 대한 전문적인 글을 작성하세요.

제목은 ##, 소목차는 ###으로 쓰고, 이외의 목차 형식은 넣지 마세요.
챕터명에 숫자를 넣지 마세요.
내용은 '입니다' 와 같은 말투로 작성하세요.

---
보고서 전체 주제: {topic}
세부 섹션 주제: {section}
''')
])
writer = writer_prompt | gpt_llm | StrOutputParser()
writer.invoke({'topic':outline.topic, 'section':outline.sections[0]})
```

**실행 결과** (섹션 1 전문):

## 정의·범위·평가 관점(멀티모달이 '무엇을' 해결하는가)

멀티모달 LLM은 텍스트만을 입력·출력으로 다루는 기존 LLM의 범위를 넘어, 이미지·오디오·비디오·문서·센서 등 서로 다른 형식의 정보를 동일한 추론 프레임 안에서 이해하고, 이를 기반으로 언어적 응답 또는 다른 모달리티의 출력을 생성하는 모델 계열입니다. 본 보고서의 연대기·기술축 중심 서술을 위해서는 "무엇이 멀티모달의 발전인가"를 먼저 고정하는 작업이 필수입니다. 즉, 입력·출력 모달리티의 확장, 해결 가능한 과업군의 확대, 그리고 정렬·추론·시간축 처리와 같은 핵심 난제의 완화 정도를 벤치마크로 계량화하는 평가 관점이 선행 정의되어야 합니다.

### 멀티모달의 정의와 범위

멀티모달이 해결하는 핵심 문제는 "비언어 정보의 구조화"와 "언어 추론과의 결합"입니다. 텍스트 중심 LLM은 언어로 기술된 세계를 강하게 모델링하지만, 실제 환경의 많은 정보는 픽셀·파형·프레임 시퀀스·레이아웃·센서 시계열로 존재합니다. 멀티모달 LLM은 이러한 비정형 신호를 언어 추론이 가능한 중간 표현으로 정렬하고, 질문응답·설명·계획·행동으로 연결하는 통합 인터페이스를 제공하는 것이 범위입니다.

또한 멀티모달은 단순히 "이미지를 입력받는 LLM"에 한정되지 않습니다. 문서(레이아웃과 텍스트가 공존), 차트·도표, 음성(대화·감정·화자 정보), 비디오(시간축 사건과 인과), 로보틱스 센서(힘·촉각·IMU·위치)까지 포함하는 것이 합리적 범위입니다. 다만 본 보고서는 멀티모달 LLM의 발전 과정을 기술축으로 비교하기 위해, 입력 모달리티 확장과 추론 능력의 결합을 중심으로 범위를 설정하고, 생성형 비디오·오디오 자체의 품질 경쟁은 부차적 축으로 다루는 것이 타당합니다.

### 텍스트 중심 LLM 대비 입력·출력 모달리티 확장

멀티모달 LLM의 확장은 입력과 출력 양쪽에서 관측됩니다.

- 이미지 입력은 시각 인식(객체·속성·관계), 공간 추론, 텍스트 포함 이미지의 판독(OCR 연계)을 통해 "보고 말하기"를 가능하게 하는 축입니다.
- 오디오 입력은 음성 인식(ASR)뿐 아니라 화자 분리, 억양·감정, 환경음 단서 등을 통해 "듣고 이해하기"를 가능하게 하는 축입니다.
- 비디오 입력은 프레임 단위 인식에서 나아가 사건의 시간적 연결, 행위 추적, 장면 전환 이해를 포함하며 "시간축 세계 모델링"을 요구하는 축입니다.
- 문서 입력은 텍스트뿐 아니라 레이아웃·표·수식·도형·주석 등 2차원 구조의 이해가 필요하며, 실제 업무 자동화와의 연결이 강한 축입니다.
- 센서 입력은 로보틱스·IoT 등에서 관측되는 시계열 신호로서, 노이즈·지연·캘리브레이션 문제와 결합되며 "행동으로 이어지는 인지"를 요구하는 축입니다.

출력 측면에서 멀티모달 LLM은 여전히 텍스트 응답이 중심이지만, 모델 계열에 따라 이미지 영역 지정(박스·마스크), 구조화된 JSON, 도구 호출 명령, 음성 합성용 텍스트, 또는 간단한 시각적 스케치 같은 비텍스트 출력까지 포함될 수 있습니다. 따라서 발전의 기준은 "입력 모달리티를 받는다"가 아니라, 그 입력을 기반으로 일관된 추론을 수행하고 과업에 유용한 형식으로 출력하는지에 놓여야 합니다.

### 대표 과업군과 요구 능력

멀티모달 LLM의 발전은 해결 가능한 과업군의 질적 변화로 측정할 수 있습니다. 주요 과업군은 다음과 같이 정리됩니다.

- VQA(Visual Question Answering)입니다. 이미지 기반 질문응답이며 인식+추론의 결합을 요구합니다. 단순 사물 인식에서 관계·상식·수리 추론으로 난도가 확장됩니다.
- 캡셔닝(Captioning)입니다. 이미지·비디오의 내용을 자연어로 요약·설명하는 과업이며, 정보 선택과 서술의 충실도가 핵심입니다.
- OCR/문서 이해입니다. TextVQA, DocVQA 계열로 대표되며, 텍스트 판독 자체와 레이아웃 기반 질의응답(표·양식·청구서)이 결합됩니다.
- 시청각 대화입니다. 이미지/비디오와 대화 맥락을 함께 유지하며 질의응답을 이어가는 과업이며, 대화 메모리와 시각 근거성(grounding)이 핵심입니다.
- 비디오 내 이벤트 추적입니다. 사건의 시작·종료·원인·결과를 시간축에서 추론하고, 객체·행위의 지속성을 유지해야 하는 과업입니다.

이 과업군들은 공통적으로 "근거 제시 가능성", "다단계 추론", "시간적 일관성", "구조화된 정보의 정확성"을 요구합니다. 따라서 단일 정확도 상승뿐 아니라, 어떤 능력이 새로 가능해졌는지에 대한 능력 기반 분해가 필요합니다.

### 핵심 난제: 정렬·추론·시간축·메모리·도구사용

멀티모달 LLM이 텍스트 중심 LLM보다 본질적으로 어려운 이유는 입력 신호의 구조가 다르고, 그 신호를 언어 토큰 공간과 연결하는 과정에서 여러 병목이 발생하기 때문입니다. 본 보고서에서는 발전을 논할 때 다음 난제를 기준축으로 둡니다.

- 정렬(alignment) 문제입니다. 시각·청각 특징이 언어적 개념과 정확히 대응되어야 하며, 객체 지시("왼쪽 위의 빨간 버튼")나 문서 내 특정 필드 지시에서 오류가 빈번합니다.
- 추론(reasoning) 문제입니다. 인식 오류가 추론 오류로 증폭되기 쉬우며, 특히 수리·논리·공간 추론에서 "보이는 것"과 "계산/규칙"의 결합이 요구됩니다.
- 시간축(temporal) 문제입니다. 비디오는 정보량이 크고 사건이 연속적이므로, 프레임 샘플링·요약·중요 구간 선택이 성능을 좌우합니다. 시간적 인과와 동시성을 이해하는 능력도 핵심입니다.
- 메모리(memory) 문제입니다. 긴 비디오, 다페이지 문서, 장시간 대화는 컨텍스트 한계를 빠르게 초과하며, 요약·검색·외부 메모리 연동이 필요합니다.
- 도구사용(tool use) 문제입니다. OCR 엔진, 검색, 계산기, 캘린더, 비디오 탐색 API, 문서 파서 등 외부 도구를 적절히 호출하고 결과를 통합하는 능력이 실제 성능을 좌우합니다. 멀티모달에서는 도구 호출의 타이밍과 입력 선택(어떤 프레임을 OCR할지 등)이 추가 난제입니다.

"발전"은 이 난제들에 대해 모델이 더 적은 휴리스틱과 더 강한 일반화로 안정적으로 해결하는 방향으로 정의될 필요가 있습니다.

### 발전을 측정하는 대표 벤치마크와 고정된 평가 기준

연대기적 비교에서 가장 중요한 것은 평가 기준의 일관성입니다. 멀티모달 LLM은 데이터·프롬프트·도구 의존성이 크므로, 벤치마크 선택과 채점 방식이 곧 발전의 정의가 됩니다. 본 섹션에서는 대표 벤치마크를 과업 성격별로 묶어 "무엇을 측정하는가"를 명확히 합니다.

- MMBench입니다. 광범위한 시각-언어 능력을 객관식 형태로 평가하여, 기본 인식·상식·추론을 폭넓게 비교하는 지표로 사용됩니다.
- SEED 계열입니다. 멀티모달 이해 전반을 포함하며, 다양한 시각적 개념과 질의 유형을 포괄하는 비교 프레임으로 활용됩니다.
- MMMU입니다. 대학 수준의 멀티모달 문제를 통해 고난도 지식·추론·도해 이해를 평가하는 축입니다. 단순 인식이 아닌 "문제 해결력"을 측정하는 데 적합합니다.
- MathVista입니다. 시각 정보(도형·그래프·표)와 수리 추론 결합을 평가하며, 멀티모달에서 흔한 계산 오류와 근거 추론 오류를 분리해 관찰하는 데 유용합니다.
- VQA 계열입니다. 역사적으로 가장 널리 쓰이는 시각 질문응답 지표이며, 모델의 기본 시각-언어 정렬과 편향 문제를 함께 드러냅니다.
- TextVQA입니다. 이미지 속 텍스트 판독과 질의응답을 결합하여 OCR 연계 능력과 오류 전파를 평가하는 데 적합합니다.
- DocVQA입니다. 문서 레이아웃 이해, 표·양식 질의응답, 다중 요소 정렬을 포함하며, 실제 업무 문서 처리 성능을 가늠하는 대표 지표입니다.

이 벤치마크들을 통해 본 보고서는 발전을 다음과 같은 고정 기준으로 측정하는 구성을 취하는 것이 바람직합니다. 첫째, 단순 정확도 상승이 아닌 과업 난도 상승(MMMU·MathVista 같은 고난도 세트에서의 개선)입니다. 둘째, 근거성 및 정렬 능력의 안정성(지시 대상 오류, 환각적 서술 감소)입니다. 셋째, 시간축·장문 컨텍스트에서의 성능 유지(비디오·다페이지 문서)입니다. 넷째, 도구사용 포함 여부에 따른 성능 분해(도구가 있어도 모델이 적절히 호출·통합하는지)입니다. 이 기준을 먼저 고정함으로써, 이후 장에서는 특정 모델이나 기법의 등장이 "무엇을 더 잘하게 했는지"를 동일한 잣대로 비교할 수 있는 토대가 마련됩니다.

---

### 실습: 5개 섹션을 `batch()`로 한 번에 작성하기

```python
writer = writer_prompt.partial(topic=outline.topic) | gpt_llm | StrOutputParser()
# topic을 미리 채워 매개변수 1개
result = writer.batch(outline.sections)
result
```

> 💡 **코드 동작 설명**: 바로 위 셀에서는 `writer.invoke()`로 섹션 1개씩 순차 생성했지만, 이번에는 **`.partial(topic=...)`으로 공통 인자(주제)를 고정한 뒤 `writer.batch(outline.sections)`로 5개 섹션을 한 번에 병렬 요청**합니다. README p.13의 "`batch()`로 병렬 실행"이 실제 성능 이점(순차 5회 호출 대신 동시 5회 요청)으로 이어지는 지점입니다. `result`는 5개의 긴 마크다운 글이 담긴 리스트입니다. (섹션 1의 내용은 위에서 이미 확인했으므로, 아래는 나머지 4개 섹션의 실제 생성 결과입니다.)

**실행 결과 — 섹션 2**:

## 초기 멀티모달 신경망에서 비전-언어 사전학습으로의 전환

### CNN·Transformer 기반 캡셔닝과 VQA의 출발점
초기 멀티모달 신경망은 이미지 이해와 언어 생성을 결합하는 과제에서 출발한 흐름입니다. 대표적으로 이미지 캡셔닝은 CNN으로 이미지를 인코딩하고 RNN(LSTM) 또는 이후 Transformer 디코더로 문장을 생성하는 방식이 주류였으며, 시각 정보는 주로 하나의 전역 특징 벡터 또는 제한된 지역 특징으로 언어 생성에 주입되는 구조입니다. VQA(Visual Question Answering) 역시 CNN 기반 시각 특징과 질문 임베딩을 결합하여 답을 분류하는 형태로 발전했으며, 이 과정에서 단순 결합(concatenation)이나 bilinear pooling 같은 융합 기법이 성능의 핵심 변수였습니다.
다만 이 시기의 접근은 과제별 지도학습 의존도가 높고, 데이터 규모와 어노테이션 품질에 성능이 크게 좌우되는 한계가 분명했습니다.

### 교차 주의 기반 비전-언어 모델의 등장과 정렬 학습의 명시화
2018년 전후로 Transformer가 언어 영역에서 보편적 백본이 되면서, 비전-언어 결합에서도 cross-attention 기반 구조가 본격화되었습니다. VilBERT, LXMERT, UNITER류 모델은 이미지의 지역 특징(대개 Faster R-CNN 계열 검출기에서 추출한 객체/영역 임베딩)과 텍스트 토큰을 입력으로 받아, 모달리티 간 상호작용을 cross-attention 또는 공동 self-attention으로 학습하는 방식입니다.

### 대규모 이미지-텍스트 사전학습과 공통 임베딩 공간의 확립
2018~2021년의 분기점은 대규모 웹 스크랩 이미지-자막 데이터로 학습하는 "대조학습 기반" 사전학습이 성능과 전이의 기준선을 재정의한 점입니다. CLIP, ALIGN류 모델은 이미지 인코더와 텍스트 인코더를 각각 학습한 뒤, 한 배치 내에서 올바른 이미지-문장 쌍은 가깝게, 다른 쌍은 멀게 만드는 대조 목적(contrastive objective)을 사용합니다.

### 학습 목표의 비교: 대조학습과 마스크드 모델링의 역할 차이
이 시기에는 대조학습 외에도 masked modeling 계열 목표가 함께 발전했습니다. UNITER류에서는 MLM(텍스트 마스킹 복원), MRM(이미지 영역/특징 마스킹 복원), ITM(이미지-텍스트 매칭 판별) 등을 결합하여 정렬과 추론 능력을 동시에 강화합니다.

### 데이터의 영향: 웹 스크랩 이미지-자막과 분포 확장의 명암
대규모 이미지-텍스트 사전학습을 가능하게 한 전제는 웹에서 수집한 이미지-자막(alt-text, 주변 문맥 등) 데이터의 양적 확대입니다. 이러한 데이터는 어노테이션 비용 없이 규모를 확장할 수 있어 개념 커버리지와 롱테일 표현 학습에 유리합니다.

### 정리: 정렬과 공통 공간이 멀티모달 전이의 중심 원리가 된 시기
초기 캡셔닝·VQA는 모달 융합을 과제 중심으로 최적화한 단계이며, 2018~2021년은 이를 범용 사전학습으로 전환한 단계입니다. 결국 "공통 임베딩 공간"과 "정렬 학습"이 멀티모달 LLM로 이어지는 발전 경로에서 성능과 확장성의 기반으로 확립된 시기입니다.

**실행 결과 — 섹션 3**:

## LLM 결합의 1세대: '비전 인코더+LLM' 어댑터 패러다임의 정립과 한계

2022~2023년의 멀티모달 LLM 결합 흐름은 "강력한 사전학습 비전 인코더의 출력을 LLM이 이해할 수 있는 입력 공간으로 옮겨 붙인다"는 어댑터 패러다임의 정립으로 요약됩니다.

### 핵심 구조: 특징 토큰을 LLM 입력 공간으로 "번역"하는 프로젝터
1세대 결합의 공통 골격은 비전 인코더가 산출한 연속 벡터(패치 토큰 또는 pooled 특징)를 "언어 토큰처럼" LLM 앞단에 주입하는 것입니다. 선형/MLP 프로젝터는 단순히 차원과 분포를 맞추는 역할을 하며, Q-Former(예: BLIP-2)은 학습 가능한 쿼리 토큰이 비전 특징에서 필요한 정보만 추출하도록 하여 LLM 입력 길이를 통제하고 정보 병목을 설계합니다.

### 대표 계열의 기술적 분기: Flamingo, BLIP-2, LLaVA의 설계 선택
Flamingo는 대규모 웹 데이터 기반의 비전-언어 사전학습을 바탕으로, 언어 모델 내부에 비전 정보를 삽입하는 교차어텐션 구조를 활용합니다. BLIP-2는 비전 인코더와 LLM을 대부분 고정하고 Q-Former로 연결부만 학습하는 전략입니다. LLaVA 계열은 "오픈 LLM+비전 인코더+프로젝터" 조합을 정렬 데이터(SFT) 중심으로 빠르게 제품화 가능한 형태로 만든 흐름입니다.

### 정렬 데이터의 진화: 이미지-설명에서 대화형 지시 데이터로
초기에는 이미지-설명(캡션) 데이터가 중심이었으나, 실제 사용자 상호작용은 질의응답·요약·비교·추론·역할극 등 "대화형 지시" 형태로 나타납니다.

### SFT와 RLHF 도입이 바꾼 UX: 정답률보다 "따르는 느낌"의 강화
SFT는 단순한 시각 질의응답 정확도 개선을 넘어, 사용자의 지시를 구조적으로 따르는 응답 형식을 학습시키는 역할을 합니다. RLHF는 "유용성, 무해성, 정중함, 과도한 확신 억제" 같은 대화 품질을 개선하는 방향으로 작동합니다.

### 환각과 세밀 인식 한계의 발생 지점: 정보 병목과 정렬의 불일치
정보 병목(특징 토큰 압축), 표현 공간 불일치(비전-언어 매핑의 근본적 어려움), 데이터와 목표의 불일치, 평가·피드백의 한계라는 네 층위에서 환각과 세밀 인식 문제가 발생합니다.

### 소결: 1세대의 의미와 다음 단계로의 동인
2022~2023년 1세대 결합은 비전 인코더와 LLM을 어댑터로 연결해 멀티모달 대화형 지시따르기를 현실적인 비용으로 구현한 전환점입니다.

**실행 결과 — 섹션 4**:

## 고도화 단계: 고해상도·문서·비디오·오디오로의 확장과 에이전트화

2023~2025년의 멀티모달 LLM 발전은 입력 양식의 확장과 함께, 모델이 "보고-읽고-듣고-행동하는" 통합 시스템으로 고도화되는 과정으로 정리할 수 있습니다.

### 이미지: 고해상도 타일링·멀티스케일, OCR 결합, 레이아웃 이해
고해상도 이미지는 단일 해상도로 전체를 축소해 입력할 경우 세부 정보가 소실되는 문제가 본질적 제약입니다. 이를 해결하기 위해 고해상도 타일링 및 멀티스케일 접근이 정착되었습니다. 문서·스크린샷 등 텍스트 중심 이미지에서는 OCR 결합이 성능을 좌우합니다.

### 비디오: 시간적 토큰 압축, 이벤트 세그먼트, 장기 컨텍스트
비디오 이해의 난점은 프레임 수가 폭발적으로 증가해 토큰 예산을 초과한다는 점과, 장면 변화·행동·인과 관계가 시간 축에 걸쳐 나타난다는 점입니다. 시간적 토큰 압축, 이벤트 세그먼트, 장기 컨텍스트 관리가 핵심 기술로 부상했습니다.

### 오디오: 음성인식·이해·생성의 통합
음성인식(ASR), 발화 의도 이해, 음성 합성(TTS)이 분리된 파이프라인에서 통합형 모델 또는 긴밀히 결합된 시스템으로 발전했습니다.

### 도구사용: 브라우징·코드·검색·OCR·ASR 호출 멀티모달 에이전트와 트레이드오프
이 단계의 가장 큰 변화는 멀티모달 LLM이 단일 모델의 "정답 생성기"에서, 외부 도구를 선택·호출·검증하는 멀티모달 에이전트로 확장된 점입니다. 모델 단독 대비 툴 결합의 핵심 트레이드오프는 성능과 비용, 그리고 신뢰성입니다.

**실행 결과 — 섹션 5**:

## 미래 과제와 신뢰성(2025~): 데이터·안전·평가의 재정의

### 데이터 패러다임의 전환: 합성데이터와 자기학습의 일상화
2025년 이후 멀티모달 LLM의 핵심 제약은 단순한 모델 규모가 아니라, 신뢰 가능한 학습 데이터의 품질·구성·권리 상태를 지속적으로 유지하는 운영 문제로 이동하는 추세입니다. 다만 합성데이터는 품질 붕괴와 편향 증폭의 위험을 내재합니다.

### 멀티모달 정렬의 고도화: 멀티모달 RL과 안전 정책의 시스템화
멀티모달 LLM의 안전은 단순히 유해 텍스트를 걸러내는 수준을 넘어, 이미지·영상·음성 입력의 해석과 그에 기반한 행동이 결합되면서 "행동 안전"의 성격을 강하게 띱니다.

### 추론의 검증 가능성: 체인오브쏘트 비노출 기반의 프로그램·툴 검증
내부 사고 과정을 그대로 출력하지 않으면서도 결과를 검증 가능하게 만드는 요구가 커지는 추세입니다. 수치 계산은 계산기로 재검산하고, 사실 확인은 신뢰 가능한 검색·지식베이스에서 인용 근거를 구조화해 첨부하는 방식입니다.

### 환각·프롬프트 인젝션·저작권/개인정보: 리스크가 비용 항목으로 계상되는 시기
멀티모달 환각은 텍스트 환각보다 실제 피해로 직결될 가능성이 큽니다. 프롬프트 인젝션은 멀티모달에서 특히 "문서 기반 워크플로"와 결합될 때 치명적입니다.

### 평가의 재정의: 벤치마크 과적합에서 업무 중심 프레임으로의 이동
멀티모달 LLM의 성능 평가는 기존 벤치마크 점수 중심 접근의 한계를 빠르게 노출하는 국면입니다. 궁극적으로 2025년 이후 멀티모달 LLM의 경쟁력은 "얼마나 똑똑한가"가 아니라 "얼마나 검증 가능하고 운영 가능한가"로 수렴하는 방향입니다.

---

### 실습: 최종 결과물 합치기

```python
draft = '\n\n'.join(result)
with open('result.md', 'w', encoding='utf-8') as f:
    f.write(draft)

print(draft[0:100])
```

**실행 결과**:
```
## 정의·범위·평가 관점(멀티모달이 ‘무엇을’ 해결하는가)

### 멀티모달 LLM의 정의와 문제 설정
멀티모달 LLM은 텍스트 기반 대규모 언어모델을 중심으로, 이미지·오디오·
```

> 💡 이 실습은 README p.19 "적은 수의 샘플을 사용해 유사한 데이터 생성하기"의 완성된 사례입니다. **`Sections`라는 최소한의 구조화된 출력(주제 1개 + 섹션 개요 5개)**만으로, `writer` 체인이 **동일한 형식 규칙("##, ### 목차 규칙, '입니다' 말투")을 공유하는 5개의 A4 반 페이지 분량 전문가 수준 글**을 생성했습니다. 사람이 5개 섹션을 일일이 쓰는 대신, **개요(적은 샘플) → 병렬 확장(batch)**이라는 2단계 파이프라인으로 하나의 완결된 보고서(`result.md`)를 자동 생성한 것입니다.

---

## Part 6. Runnables 심화 실습

### 이론: 특수한 Runnables (p.17)

> 값을 직접 저장하지 않고, `Runnable`들을 연결하는 역할의 `Runnable`
> - `RunnablePassthrough()`: 직전 `Runnable`의 출력을 그대로 Return
> - `RunnableParallel()`: 1개 이상의 `Runnable`을 실행하고, 그 결과를 Dict로 Return
> - `.assign()`: `Runnable` 값을 전달하고, 그 결과를 가져와 결합
>
> LangChain의 한계점인 Sequential 구성을 보다 다양화 — 이후 LangGraph 구조에서 SeqGraph 구조로 더욱 확장

노트북은 `RunnableSequence`(기본 체인 구조)를 소개하며, 이번 Part의 7가지 실습으로 위 이론을 하나씩 검증합니다.

### 실습 1: `RunnablePassthrough` — 직전 출력을 그대로 전달

```python
from langchain_core.runnables import RunnablePassthrough

prompt1 = ChatPromptTemplate(["{director}의 대표 작품은 무엇입니까? 하나의 작품만 선택하고, 해당 작품에 대해 20자 이내로 설명하세요."])
chain1 = (
    prompt1
    | gpt_llm
    | StrOutputParser()
    | {'answer': RunnablePassthrough()})

response = chain1.invoke("봉준호")
response
```

**실행 결과**: `{'answer': '**기생충**: 계층 갈등을 그린 블랙코미디'}`

`StrOutputParser()`가 반환한 문자열을 `RunnablePassthrough()`가 **그대로** `{'answer': ...}` 딕셔너리의 값으로 감쌉니다 — 값 자체를 바꾸지 않고 "포장"만 바꾸는 역할입니다.

### 실습 2: `RunnableParallel` — 여러 체인을 동시 실행해 Dict로 반환

```python
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate(["색깔을 하나 알려주세요, 색깔만 출력하세요."])
prompt2 = ChatPromptTemplate(["음식을 하나 알려주세요, 음식만 출력하세요."])

chain1 = prompt1 | gpt_llm | StrOutputParser()
chain2 = prompt2 | gpt_llm | StrOutputParser()

chain3 = RunnableParallel(color = chain1, food = chain2)
# 개별 체인을 병렬 실행한 뒤, dict로 반환

chain3.invoke({})
```

**실행 결과**: `{'color': '파란색', 'food': '비빔밥'}`

서로 무관한 두 체인(`chain1`, `chain2`)이 **병렬로 동시에 실행**되어 하나의 Dict로 합쳐집니다 — 두 체인이 서로의 결과에 의존하지 않으므로 순차 실행보다 빠릅니다.

### 실습 3: `.assign()` — 이전 결과를 다음 체인에 전달하며 결합

> `RunnableParallel`을 사용하면 중간 체인의 결과를 전달하여, 다음 체인의 결과를 함께 얻을 수 있습니다.

```python
prompt1 = ChatPromptTemplate(["잭슨빌은 어느 나라의 도시입니까? 나라 이름만 출력"])
prompt2 = ChatPromptTemplate(
    ["{country}의 대표적인 인물 3명을 나열하세요. 인물의 이름만 출력하세요."]
)

chain1 = prompt1 | gpt_llm | StrOutputParser()
chain2 = prompt2 | gpt_llm | StrOutputParser()

chain3 = RunnableParallel(country = chain1).assign(people = chain2)
#        {    'country': '미국'           }  +    {'people': chain2(미국)}

chain3.invoke({})
```

**실행 결과**: `{'country': '미국', 'people': '조지 워싱턴  \n에이브러햄 링컨  \n마틴 루터 킹 주니어'}`

**Part 4·5의 `RunnableParallel().assign()`이 왜 그렇게 동작했는지가 여기서 명확해집니다.** `RunnableParallel(country=chain1)`이 먼저 `{'country': '미국'}`을 만들고, `.assign(people=chain2)`가 **이 딕셔너리 전체를 `chain2`(`{country}` 변수를 요구)에 입력**하여 결과를 `people` 키로 추가합니다. `chain1`(도시→국가)이 먼저 실행되고, 그 결과가 `chain2`(국가→인물)의 입력이 되는 **순차 의존 관계**가 `.assign()` 하나로 표현됩니다.

### 실습 4: `RunnablePassthrough.assign()` — 새로운 매개변수 추가하기

> chain2에서 새로운 매개변수가 추가되는 경우는 어떻게 해야 할까요?

```python
prompt1 = ChatPromptTemplate([
    "{city}는 어느 나라의 도시인가요? 나라 이름만 출력하세요."])
prompt2 = ChatPromptTemplate([
    "{country}의 유명한 인물은 누가 있나요? {num} 명의 이름을 나열하세요. 사람 이름만 ,로 구분하여 나열하세요."])

chain1 = prompt1 | gpt_llm | StrOutputParser()

chain2 = (
    RunnablePassthrough.assign(country = chain1)
    # 입력받은 city, num에 country를 추가하여 전달
    # city, num          +    country

    | prompt2
    # country, num을 받아 실행
    | gpt_llm
    | StrOutputParser()
)

print(chain2.invoke({"city": "잭슨빌", "num": "3"}))
```

**실행 결과**: `조지 워싱턴, 에이브러햄 링컨, 마틴 루터 킹 주니어`

실습 3의 `RunnableParallel(country=chain1)`은 **입력값(city, num)을 버리고 `country`만 남기지만**, `RunnablePassthrough.assign(country=chain1)`은 **입력값 전체(`city`, `num`)를 유지한 채 `country`만 추가**합니다. `prompt2`가 `country`뿐 아니라 `num`도 필요로 하므로, 원본 입력을 보존하는 이 방식이 필수적입니다 — `RunnableParallel(...)`과 `RunnablePassthrough.assign(...)`의 차이가 실제 결과로 명확히 드러나는 대목입니다.

### 실습 5: `.assign()` 체이닝 — 여러 개 연속 연결

> `assign`을 여러 개 연결할 수 있습니다.

```python
chain4 = (prompt2
    | gpt_llm
    | StrOutputParser())

chain3 = RunnablePassthrough.assign(country = chain1).assign(people = chain4)
#        {'city', 'num'}      +    {'country'}         +    {'people'}

chain3.invoke({"city": "부에노스 아이레스", "num": "3"})
```

**실행 결과**:
```python
{'city': '부에노스 아이레스',
 'num': '3',
 'country': '아르헨티나',
 'people': '리오넬 메시, 디에고 마라도나, 에바 페론'}
```

`.assign()`을 두 번 연쇄하여 `city/num → +country → +people`으로 **딕셔너리가 단계적으로 누적**됩니다 — Lab 6·7의 `RunnableParallel(input=...).assign(query=...).assign(result=...).assign(answer=...)` 패턴이 바로 이 원리의 확장입니다.

### 실습 6~7: JsonOutputParser와 `.assign()`의 결합

```python
prompt1 = ChatPromptTemplate(
    ["영화 배우 한 명과 대표작 하나를 출력하세요. json 형식으로 출력하고, 각 항목은 actor, movie로 표시하세요."])
prompt2 = ChatPromptTemplate(["{actor}는 {movie}에서 어떤 역할을 했습니까?"])

chain1 = prompt1 | gpt_llm | JsonOutputParser()
# {actor, movie}
chain2 =(
     chain1 | prompt2 | gpt_llm | StrOutputParser()
)
chain2.invoke({})
```

**실행 결과**: `'송강호는 영화 **〈기생충〉(2019)**에서 **김기택** 역을 맡았습니다.  \n김기택은 기우(최우식)·기정(박소담)의 아버지이자 김가족의 가장으로, 박사장(이선균) 집에 **운전기사로 취업**하며 이야기를 이끌어가는 핵심 인물입니다.'`

```python
chain3 = prompt2 | gpt_llm | StrOutputParser()

chain4 = chain1.assign(result = chain3)

chain4.invoke({})
```

**실행 결과**:
```python
{'actor': '송강호',
 'movie': '기생충',
 'result': '송강호는 영화 **〈기생충〉(2019)**에서 **기택(김기택)** 역을 맡았습니다.  \n가난한 가족의 **아버지**로, 박사장(이선균) 집에 **운전기사로 취업**하면서 가족들이 차례로 그 집에 들어가게 되는 과정의 중심에 있는 인물입니다.'}
```

> 💡 `chain2`는 `chain1`(JSON 출력)에 `prompt2 | llm | parser`를 **파이프로 바로 연결**하여 **최종 답변 문자열만** 반환합니다. 반면 `chain4`는 `chain1.assign(result=chain3)`으로 연결하여 **원본 JSON(`actor`, `movie`)에 `result`를 추가한 전체 딕셔너리**를 반환합니다 — Lab 4의 `rag_chain`(답변만 반환) vs `rag_chain_with_source`(근거+답변 모두 반환)와 동일한 설계 선택입니다. 두 방식 모두 유효하지만, **디버깅이나 근거 추적이 중요한 경우 `.assign()`으로 중간 결과까지 보존**하는 쪽이 유리합니다.

---

## Part 7. 정리

### 7.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | `JsonOutputParser()` (스키마 없음) | `get_format_instructions()`가 `"Return a JSON object."` 한 문장뿐 → 실행마다 스키마가 달라짐을 실제로 확인 |
| 2 | `JsonOutputParser(pydantic_object=Recipe)` | 형식 지시가 전체 JSON Schema로 확장되어, 재료가 바뀌어도 6개 필드 스키마가 고정됨 |
| 3 | `.partial(instruction=...)` | 매번 넘기던 고정 인자를 프롬프트에 미리 채워, 체인 재사용성 향상 |
| 4 | `with_structured_output(Recipe)` | API `response_format`으로 스키마를 강제, Pydantic 객체 직접 반환 |
| 5 | `rich.print()` 결과 인코딩 손상 | Windows 콘솔 cp949 + rich 렌더링 조합에서 실제로 발생한 인코딩 이슈 |
| 6 | `PydanticOutputParser` | `with_structured_output` 미지원 모델의 대안, 동일하게 Pydantic 객체 반환 |
| 7 | `outliner`(Sections) → `writer.batch()` | 5개 섹션 개요 → 5개 전문 보고서 섹션 병렬 생성 → `result.md`로 합치기 |
| 8 | `RunnablePassthrough` | 직전 출력을 그대로 포장 |
| 9 | `RunnableParallel` | 독립적인 두 체인을 병렬 실행 → Dict 결합 |
| 10 | `.assign()` (RunnableParallel 기반) | 이전 결과를 다음 체인 입력으로 전달하는 순차 의존 표현 |
| 11 | `RunnablePassthrough.assign()` | 원본 입력을 보존하며 새 필드 추가(`RunnableParallel`과의 핵심 차이) |
| 12 | `.assign()` 체이닝 | 여러 단계를 거치며 딕셔너리에 필드를 누적 |

### 7.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 확인한 것 |
|---|---|
| LangChain의 Structured Output(p.16) | JSON 파서(스키마 없음) → Pydantic 파서(스키마 고정) → `with_structured_output()`(API 강제) 3단계로 구조화 수준이 높아지는 과정을 실제 출력 차이로 확인 |
| 특수한 Runnables(p.17) | `RunnablePassthrough`/`RunnableParallel`/`.assign()` 7가지 실습으로 각각의 정확한 동작(포장/병렬/순차결합/입력보존) 검증 |
| Lab #3 "적은 수의 샘플을 사용해 유사한 데이터 생성하기"(p.19) | `Sections`(5개 개요) → `writer.batch()`(5개 섹션 병렬 작성) → `result.md` 합치기 |
| Runnable과 Invoke — `batch()`(p.13) | `writer.batch(outline.sections)`로 5개 섹션 동시 생성 |

### 7.3 참고 자료

- LangChain Output Parsers: <https://python.langchain.com/docs/concepts/output_parsers/>
- Pydantic 공식 문서: <https://docs.pydantic.dev/>
- `with_structured_output()` 가이드: <https://python.langchain.com/docs/how_to/structured_output/>
- Rich (Python 콘솔 출력 라이브러리): <https://github.com/Textualize/rich>
- 원본 실습 노트북: `[강의내용]_3_LangChain을_이용한_데이터_생성과_처리.ipynb`

### 7.4 다음 단계

- **Lab 2** (README p.18, "LangChain을 이용한 데이터 분류와 전처리")에서는 이번 실습의 구조화된 출력을 **분류(classification) 작업**에 적용합니다 — "분류 프롬프트 구성 → 분류 체인 → 결과 평가 및 성능 개선"의 흐름으로, 이번 실습의 Pydantic 스키마 고정 기법이 "카테고리 값이 미리 정한 목록 중 하나임을 보장"하는 데 그대로 쓰입니다.
- **Lab 4** (`[실습]_4_벡터_데이터베이스_기반_RAG_어플리케이션_만들기.ipynb`)에서는 이번 Part 6에서 배운 `RunnableParallel().assign()`이 RAG 체인에서 "답변과 근거(context)를 함께 반환"하는 데 실제로 활용됩니다.
- 실무 팁: 프로토타이핑 단계에서는 Part 2의 스키마 없는 `JsonOutputParser`로 빠르게 실험하고, 프로덕션에 반영할 때는 Part 3~4처럼 Pydantic 스키마로 반드시 고정하는 것을 권장합니다 — 이번 실습에서 실제로 확인했듯, 스키마가 없으면 동일한 프롬프트도 입력에 따라 다른 구조의 응답을 만들어낼 수 있습니다.
