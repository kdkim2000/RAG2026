# 2026년도 — RAG 개발 심화

> 본 문서는 `교재_0803.pdf` (2026년도 삼성SDS AI 교육 「RAG 개발 심화」 강의 교재, 총 82페이지)를 마크다운으로 변환/정리한 것입니다. 슬라이드의 제목·본문·도식·예시·출처를 누락 없이 옮기고, 원문에는 없던 문맥 설명을 최소한으로 덧붙였습니다. 원본 슬라이드의 페이지 번호를 각 절 제목 옆에 `(p.N)`로 표기했습니다.
>
> **업데이트 내역**: ① GPT Builder 챕터(원문 p.76~82)는 더 이상 사용하지 않는 기술로 판단되어 제외 ② 수업 중 필기(RAG의 다양한 시나리오, Embedding 모델, RAGAS 상세)를 관련 챕터에 "(수업 필기)" 표시로 추가.

**강사**: 변형호 ([GitHub: NotoriousH2](https://github.com/NotoriousH2))

## 과정 개요 (p.3)

**과정 목표**
- LLM의 개념을 이해하고, 대표적인 활용 사례를 학습한다.
- 랭체인을 비롯한 다양한 라이브러리를 학습하고, GPTs, Action 등의 기능을 실습한다.

**과정 구성**
1. LangChain 학습 (Day 1)
2. RAG 심화와 LangChain 개발 (Day 2)

### Day 1. LangChain 학습 (p.4)
1. LangChain 컴포넌트
2. LangChain으로 자동화 및 챗봇 앱 만들기
3. 벡터스토어와 RAG

### Day 2. RAG 심화 – LangChain 개발 (p.5)
1. RAG의 다양한 성능개선 방법
2. RAG의 성능평가 방법
3. LangChain 실용 예제


---

# Day 1. LangChain 학습

## 1. LangChain 컴포넌트 소개 (p.6)

### LangChain (p.7)
- [https://www.langchain.com/](https://www.langchain.com/)
- LLM 어플리케이션 개발에 필요한 다양한 도구 지원
  - 높은 추상화(Abstraction) 기반의 쉬운 구현 — "LLM Application Toolbox"
- 자세한 공식 문서 및 다양한 예제 코드 존재
  - 바이브 코딩 구현도 쉬움

### LangChain 구조 (p.8)
모듈식의 구성 요소 연결
- 모델, 프롬프트, 체인
- 데이터베이스, 메모리
- 툴 사용과 에이전트

원문의 아키텍처 다이어그램(레이어 구조)은 다음과 같이 구성되어 있습니다.

- **Protocol**: `LangChain Core` — LCEL(LangChain Expression Language): Parallelization · Fallbacks · Tracing · Batching · Streaming · Async · Composition (Python / JavaScript)
- **Integrations / Components**: `LangChain Community` — Models I/O(Model, Prompt, Example Selector, Output Parser), Retrieval(Retriever, Document Loader, Vector Store, Text Splitter, Embedding Model), Agent Tooling(Tools, Toolkits)
- **Cognitive Architectures**: `LangChain`(Chain/Agents/Advanced Retrieval Strategies), `Templates`(Reference Applications)
- **Deployment**: `LangServe` — Chains as APIs
- **Observability**: `LangSmith` — Monitoring, Feedback, Evaluation, Annotation, Playground, Testing, Debugging

### LangChain과 LLM 모델 연결하기 (p.9)
- OpenAI 모델 목록: [platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- Google 모델 목록: [ai.google.dev/gemini-api/docs/models?hl=ko](https://ai.google.dev/gemini-api/docs/models?hl=ko)
- HuggingFace 공개 모델 목록: [huggingface.co](https://huggingface.co)
- Ollama 공개 모델 목록: [ollama.com/models](https://ollama.com/models)
- 모델별 성능/가격 비교: [LLM 지능 리더보드 (Artificial Analysis)](https://artificialanalysis.ai/leaderboards/models)

### LangChain + LLM Provider (p.10)
- LLM Provider별 별도의 라이브러리로 빠른 업데이트
  - `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`
- 오픈 모델 및 임베딩 모델 연결
  - `langchain_huggingface`, `langchain_ollama`
  - [huggingface.co](https://huggingface.co)

### LangChain 1.0 (2025.10) (p.11)
- 기존 LangChain 라이브러리의 분리 및 개편
  - `langchain`: LLM Agent 개발을 위한 모듈 위주
  - `langchain-classic`: 기존의 다양한 도구들
- 기존 코드가 돌아가지 않는 경우
  - `langchain_classic`으로 수정하면 대부분 해결됨
  - [migrate/langchain-v1 공식 가이드](https://docs.langchain.com/oss/python/migrate/langchain-v1)

### Chat Models in LangChain (p.12)

여러 Provider의 Chat Model을 동일한 방식으로 초기화하고 호출하는 예시입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    temperature=1,
    max_tokens=8192,
    thinking_budget=1
)

gpt5 = ChatOpenAI(
    model='gpt-5.2',
    reasoning_effort='low',
    verbosity='medium'
)

gpt5.invoke("안녕? 너는 누구니?")
# AIMessage(content='안녕하세요. 저는 OpenAI가 만든 ... 언어 정리, 공부 도움, 코딩 같은 작업을 도와드릴 수 있어요...',
#           additional_kwargs={'refusal': None},
#           response_metadata={'token_usage': {'total_tokens': 87, 'completion_tokens_details': {...}, ...},
#                               'model_provider': 'openai', ...},
#           id='lc_run--...')
```

### Runnable과 Invoke (p.13)
- LangChain의 구성요소는 `Runnable` 클래스
  - `Runnable` 계열의 클래스는 `invoke()`를 통해 동기 실행, `ainvoke()`로 비동기 실행
  - `batch()`로 병렬 실행, `abatch()`로 병렬 비동기 실행
- `init_chat_model()`을 통한 API 통합 연결
  - `model_provider`, `model_name` 전달
  - `SystemMessage`, `HumanMessage`, `AIMessage` 형태로 메시지 클래스 입력

### [실습] LangChain 기본 구조 (p.14)
- 기본 프롬프트 구성 방법 이해하기
- 다양한 LLM 모델과 LangChain 연결하기
- 병렬 실행을 위한 `batch`, 순차적 출력을 위한 `stream` 사용하기
- 모델의 프롬프트 템플릿과 LLM 연동하기

## 2. LangChain으로 자동화 및 챗봇 앱 만들기 (p.15)

### LangChain의 Structured Output (p.16)
- 기존의 LLM은 단순 Message 리스트를 전달하는 방식
- 구조화된 출력은 별도의 후처리 필요 없이 DB/프롬프트에 바로 연결할 수 있음
  - LLM 프로그램 구조 고도화에서 필수적
- 다양한 파서를 통한 출력 형식 변환
  - JSON, Pydantic, DateTime, ...
- LLM의 `with_structured_output()` 기능 활용

### 특수한 Runnables (p.17)
값을 직접 저장하지 않고, `Runnable`들을 연결하는 역할의 `Runnable`
- `RunnablePassthrough()`: 직전 `Runnable`의 출력을 그대로 Return
- `RunnableParallel()`: 1개 이상의 `Runnable`을 실행하고, 그 결과를 Dict로 Return
- `.assign()`: `Runnable` 값을 전달하고, 그 결과를 가져와 결합

LangChain의 한계점인 Sequential 구성을 보다 다양화
- 이후 LangGraph 구조에서 SeqGraph 구조로 더욱 확장

### [실습] LangChain을 이용한 데이터 분류와 전처리 (p.18)
- 분류 프롬프트 구성하기
- LangChain 검색 모듈과 분류 체인 연결하기
- 분류 결과 평가하고 성능 개선하기

### [실습] LangChain을 이용한 데이터 생성과 처리 (p.19)
- LLM의 구조화된 출력을 사용하여, 다양한 데이터 전처리하기
- 적은 수의 샘플을 사용해, 유사한 데이터 생성하기

## 3. 벡터스토어와 RAG (p.20)

### LLM 모델의 한계 (p.21)
- 언어의 패턴을 이해하여, 가장 그럴듯한 답변을 제공하는 모델
  - 다양한 데이터에 대한 사전학습(Pretrain)을 통해 지식과 언어적 패턴 이해
  - 높은 유연성과 활용성
- 높은 수준이 항상 정확성을 의미하지는 않음
  - 최신 데이터 / 내부 데이터 / 도메인 특화 데이터에 대한 질문이라면?
  - 정확한 답변을 하지 못하는 Hallucination(환각)의 위험이 큼

### 검색증강생성 — Retrieval Augmented Generation (RAG) (p.22)
- LLM과 정보 검색의 결합
  - 질의와 관련된 정보를 검색하여 프롬프트에 함께 전달
  - LLM의 In-Context Learning 능력 기반

### RAG 프롬프트 예시 (p.23)
정보와 질문을 구분하여 제공 — 정확한 정보의 검색이 무엇보다 중요

```
다음의 정보를 참고하여, 질문에 답하세요.
---
정보:
2025년 5월 13일, 삼성SDS는 웨스틴조선서울에서 열린 '금융의
미래를 선도하기 위한 삼성SDS Industry Day'에서 발전하고 있는
금융분야의 환경에 대응하기 위해, 금융업무에 생성형 AI를 적용한
사례를 공유했습니다.
---
질문:
삼성SDS Industry Day는 언제 열렸나요?
```

### RAG의 Retrieval: 검색 (p.24)
정확한 맥락을 제공하려면?
- 데이터베이스 내용 중 질문과 관련 있는 내용이 무엇인지를 알아야 함

관련성을 측정 가능한 형태로 수치화하기
- 텍스트를 벡터로 변환하는 Embedding/Indexing 방법 필요

> **도식 설명**: 데이터베이스에 「양식 요리책」, 「애견 훈련법」, 「RAG 심화 강의자료」 3개의 문서가 있을 때, "2025년 5월 13일에 열린 행사의 이름은?"이라는 질문이 들어오면 관련성이 높은 「RAG 심화 강의자료」만 정확히 찾아내야 함을 보여주는 예시입니다.

### RAG의 5-step Process (p.25)
1. **Indexing** — 데이터베이스 사전 구성
2. **Processing** — 입력 쿼리를 검색이 잘 되도록 전처리
3. **Searching** — 질문과 가장 적합한 데이터 검색하여 나열
4. **Augmenting** — 쿼리와 Searching 결과를 포함하는 프롬프트 구성
5. **Generating** — LLM에 전달해 답변 생성

### Indexing: 데이터 준비하기 (p.26)
- LLM의 Context는 한정적이므로, 전체 문서를 입력하는 것은 효과적이지 않음
  - 여러 문서가 포함된 경우에도, 필요한 부분만 선택적으로 검색할 수 있도록 분리
  - 예) 페이지/문단 단위로 나누어 저장
- **Chunking(청킹)**: 전체 텍스트 코퍼스를 분할하는 작업
  - 주로 글자수/토큰수/챕터 등을 활용하여 분할
  - 최적 청킹 기법 / 청크 크기는 도메인/데이터마다 매우 다를 수 있음
    - OpenAI의 기본 RAG 구조: 800 Chunk / 400 Overlap
  - LangChain의 `RecursiveCharacterTextSplitter()` 등의 모듈 사용

### Indexing: 적절한 청크 사이즈 선택하기 (p.27)
- 청크 크기가 작은 경우 — 주변 정보 활용 어려움
- 청크 크기가 큰 경우 — 불필요한 정보 포함, 임베딩의 정확도 감소
- 청크 오버랩 — Context의 의미 보존

> **원문 노트(제목: RAG의 이해)**: "RAG를 활용하면, LLM이 사용자의 입력에 대해 보다 정확한 답변을 생성하도록 유도할 수 있습니다. 그 이유는 프롬프트 내에 데이터베이스의 내용을 포함하여 이를 활용할 수 있기 때문입니다. 데이터베이스는 청크 단위로 저장되며, 적절한 청크 사이즈를 찾는 것이 중요합니다."

### Indexing: 청킹에 대한 견해 변화 (p.28)
청킹을 항상 해야 할까?
- 모델의 Context Length/Long Context 처리 능력과 데이터를 보고 결정
  - (참고) AI 논문 한편: 10k VS GPT-5.6 Context 1M
- **2026: LLM의 장문맥 처리 능력이 향상됨**
  - 문서가 크지 않다면(100k 미만), 작게 분할하는 대신 Full Context를 넣는 것이 더 안정적
  - 문서가 크다면, 큰 청크를 통해 맥락 단절을 방지함
    - Q) 큰 청크는 검색이 잘 안 되는데? → 큰 청크를 효과적으로 검색하기 위한 방법 고려

### LangChain Document Loader (p.29)
다양한 형식의 파일이나, API 서비스를 연결하여 데이터를 불러오는 구조
- [python.langchain.com/docs/integrations/document_loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- PDF, DOCX, XML 등의 파일 로더
- Arxiv, Tavily, Pubmed 등의 API 로더

### Docling: 구조화된 데이터 추출에 강한 파서 (p.30)
- [docling-project.github.io/docling](https://docling-project.github.io/docling/)
- PDF 파일 → JSON/Markdown 변환
- 이미지/테이블 추출
- GPU 실행 추천, 또는 [Docling-Serve](https://github.com/docling-project/docling-serve)를 통한 API 서빙 가능

### Processing: 사용자 쿼리 처리하기 (p.31)
- 의도 분류 / 작업 선택 / 쿼리 재구성
  - 어떤 DB로 검색할 문제인지를 사전에 파악
  - LLM을 통한 쿼리 수정
- 짧은 쿼리의 전처리
  - 의미 이해를 위한 정보 불충분
  - 추가 정보 / 이전 대화를 고려하여 Contextualize
- 긴 쿼리의 전처리
  - 불필요한 정보 포함
  - 요약하거나 일부 내용 삭제

> **도식 예시**: "[상품명] 교환 문의드립니다. 어떻게 하면 될까요?" → (재구성) → "제품 환불에 대한 방법을 문의드립니다. 어떻게 진행하면 되나요?"

### Searching: 쿼리와 적합한 데이터 검색하기 (p.32)
**Dense Retrieval / Sparse Retrieval**
- Embedding 기반의 Semantic 검색 (DENSE)
  - 주로 트랜스포머 계열 모델을 활용하여, 텍스트를 Dense 벡터로 변환
  - 정확하게 일치하지 않아도 유사한 의미를 탐색
- Keyword 기반의 Lexical 검색 (SPARSE)
  - BM25, TF-IDF, SPLADE 등의 인덱싱 기법을 사용해, 키워드의 포함 여부로 인덱싱
  - 정확하게 일치하는 경우에 높은 가중치
- 각각의 검색은 언제 필요할까?

### Searching: 벡터데이터베이스 = 임베딩 저장 공간 (p.33)
벡터 형태의 데이터 저장 및 검색을 제공하는 소프트웨어
- 비정형 데이터를 임베딩으로 변환하고, 이를 저장/검색
- 트랜스포머 기반 기존 Text Embedding 기술 활용
- 자연어, 그래프, 이미지 등의 데이터가 많아지면서 중요도 증가
  - LLM뿐만 아니라 데이터베이스 자체로도 활용

### Embedding 모델 (수업 필기)

임베딩 모델도 아래와 같이 세대가 구분됩니다.

1. **(~2024) BERT 기반의 모델**: 문맥상 단어/토큰 임베딩의 '평균'으로 문장 임베딩을 구성
   - 예: BGE-M3, OpenAI Embedding
   - 예: [Microsoft Harrier](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) — 기존 임베딩 모델을 파인튜닝한 모델
2. **(2025~) LLM 기반의 모델**: LLM이 이해한 문맥값 자체를 임베딩으로 사용
   - 예: Qwen3 Embedding
- 임베딩 모델 성능 비교: [MTEB 리더보드](https://huggingface.co/spaces/mteb/leaderboard)

### Searching: 벡터DB 활용 예시 (p.34)
검색: 질문(Query)과 청크(Chunks) 비교
- DB에 각 청크들의 임베딩을 저장
- 질문이 입력되면, 질문의 임베딩과 청크들의 임베딩 유사도 계산
- Top K Chunks를 Return

청크 검색 후, Task에 따른 활용
- **RAG**: LLM 프롬프트에 추가하여 답변 생성
- **Recommendation**: 해당 상품 혹은 문서 전달

### Searching: 대표적 Vector Database (p.35)
Vector 검색 전용 DB
- **Pinecone**: 클라우드 기반의 유료 서비스
- **Milvus, Qdrant, Chroma, Weaviate**: Online/Self Host 사용 지원

기존 DB(Elastic Search, OpenSearch) 등의 벡터 검색 지원
- 벡터 수가 많지 않고, 기존 DB를 운영한다면 기존 DB에서 활용 권장

### Searching: Vector Database의 Metric Types (p.36)
- **Euclidean Distance (L2 Distance)** — 벡터 간의 직선 거리
- **Cosine Distance** (임베딩이 Normalized인 경우) — 1 − Cosine Similarity
- **MMR (Maximum Marginal Relevance Search)** — 유사성과 다양성의 균형을 고려한 방법
  - `Alpha * 쿼리와의 유사성 – (1-Alpha) * (Context 중 가장 가까운 청크와의 유사성)` 순으로 검색

### [실습] 벡터데이터베이스 기반 RAG 어플리케이션 (p.37)
- `WebBaseLoader()`를 통한 웹페이지 내용 기반의 RAG
- ChromaDB를 이용하여 뉴스 기사 청크 저장 및 검색

### [실습] PDF 내용 기반 질의응답 어플리케이션 (p.38)
- PDF 파일의 내용을 기반으로 연속적 질의응답 수행하기
- 요약 알고리즘의 stuff, map-reduce, refine 과정 수행하기

---

# Day 2. RAG 심화 – LangChain 개발 (p.39)

## 1. RAG의 다양한 성능개선 방법 (p.40)

### RAG 파이프라인의 성능 개선하기 (p.41)
RAG 파이프라인에서의 모든 요소는 성능에 큰 영향
- **Chunking**: Size / Top K
- **Searching**: Semantic / Lexical

그 외의 의사결정
- 정적인 Chunking은 맥락 손실이 큰데, 어떻게 보완할까?
- 검색 Chunk의 순서는 어떻게 할까?
- Semantic/Lexical의 비중은 어떻게 할까?
- 단일 검색만으로 해결되지 않는 문제는 어떻게 해야 할까?

### Indexing: Small2Big Chunking / Sliding Window (p.42)
검색 정확도를 위해 청크의 사이즈를 최대한 작게 구성하고, 검색 이후 해당 청크의 주변 정보 포함

- **Big Chunk/Small Chunk의 2단 구성**
  - 작은 청크로 검색, 연결된 큰 청크 전달
  - LangChain: `ParentDocumentRetriever`
- **Sliding Window**
  - Small Chunk의 위치 기준으로 앞뒤 문맥 포함

> **원문 노트(제목: R-G Decoupling(디커플링))**: "텍스트의 길이가 길어질수록 임베딩 알고리즘은 해당 내용을 모두 포함하게 되므로 검색에서의 정확도가 감소할 수 있습니다. **이를 해결하기 위한 Small2Big Chunking/Sliding Window**는 작은 크기의 청크로 검색을 하고, 실제 Augmenting 단계에서는 큰 청크를 포함하는 방법입니다. 예를 들어, (중략) 이 방법은 임베딩의 효과적 검색과 긴 Context를 모두 활용할 수 있는 방법으로 활용되고 있습니다." *(원문에 "(중략)"으로 예시가 생략되어 있음)*

### Contextual Retrieval (p.43)
청킹의 앞부분 단절을 완화하기 위해, 전체 맥락을 고려해 청크의 앞부분을 추가
- 키워드 없음 / 상대적 표현 / 앞부분 분절과 같은 부정확한 청킹 완화 효과
- Context Caching을 통해 비용을 절약할 수 있음 (오픈 모델 사용도 가능)
  - Claude의 보고서에서, Hybrid 검색 성능 대폭 향상

> **도식 예시**: 원본 청크 "이 청크는 A회사의 LLM 기반 노트 작성 앱에 대한 2026년 보고서의 내용입니다. 2025년 이용자 수는 OO명이었습니다. [기존 청크 위치]" → (헤더 요청: 전체 문서 + 청크를 LLM에 전달) → 보강된 청크 "이용자 수가 전년 대비 37% 증가하며, 해당 서비스의 성공적인 시스템 통합을 입증했습니다." 가 앞부분에 추가됨.

### Advanced RAG (Processing) (p.44)
- **Multi-Query Retrieval**
  - 쿼리를 여러 개의 버전으로 세분화 후 각각의 검색을 모두 수행
  - 쿼리의 검색 다변화
  - LangChain에 구현되어 있으나, 직접 구현해도 됨
- **Query Reconstruction**
  - 추가 맥락을 고려해, 쿼리를 적절하게 재구성
  - 단순 변환이 아니라, 추가 정보가 필요할 수 있음

> **도식 예시**: "[상품명] 교환 문의드립니다. 어떻게 하면 될까요?" → "환불 조건을 알려주세요." → "바꿔줘" 와 같이 짧고 모호한 쿼리들을 "제품 환불에 대한 방법을 문의드립니다. 어떻게 진행하면 되나요?" 처럼 재구성.

### HyDE: Hypothetical Document Embedding (p.45)
- 사용자의 질문은 특정 용어나 세부사항이 포함되지 않는 경우가 있음
  - LLM을 이용해, 질의에 대한 가상의 답변이나 관련 문서를 생성
  - 해당 문서를 통해 다시 검색하면, 질의로 검색하는 것에 비해 정확도가 높다는 결과
    - 이후 RAG 파이프라인에 연결
- 출처: *Precise Zero-Shot Dense Retrieval without Relevance Labels* — [aclanthology.org/2023.acl-long.99.pdf](https://aclanthology.org/2023.acl-long.99.pdf)

### Advanced RAG (Searching) (p.46)
- **Hybrid Search**
  - Lexical, Semantic 두 검색 결과를 모두 고려
  - 주로 Rank의 조화평균(Reciprocal Rank) 사용
  - (도식) Lexical(구조적 유사도) 50% + Semantic(의미적 유사도) 50% → 최종 결과
- **Metadata Filtering**
  - 전처리 과정에서 메타데이터 부여 (LLM 사용 가능)
  - 메타데이터를 통해 관련 정보만 선택 — 카테고리, 페이지, 날짜, 작성자 등

### Advanced RAG (Augmenting) — Long Context (p.47)
Long Context Model의 시대
- 과거의 모델 Context: 16k (GPT-3.5)
- 전보다 많은 데이터를 처리 — RAG에 넣을 수 있는 Context도 증가
  - Larger K, Bigger Chunk

### Advanced RAG (Augmenting) — Lost-in-the-middle Error (p.48)
- 정답이 포함되어 있음에도 불구하고, 답변의 성능이 떨어지는 현상
  - 앞뒤의 Context가 노이즈로 작용
  - 긴 맥락 이해 능력 부족도 영향
- 출처: *Lost in the Middle: How Language Models Use Long Contexts* — [arxiv.org/pdf/2307.03172](https://arxiv.org/pdf/2307.03172)

### Advanced RAG (Augmenting) — Context Optimization (p.49)
좋은 답변을 생성하기 위한 Context Optimization이 필요
1. 중요 정보의 위치 조절
2. 불필요한 청크 제거

**Question**: 청크의 중요성을 어떻게 판단할까?

출처: *Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG* — [arxiv.org/abs/2410.05983](https://arxiv.org/abs/2410.05983)

### Advanced RAG (Augmenting) — Reranking (p.50)
- 질문과 문서의 관련도를 판별하여 재정렬 또는 필터링
- 임베딩 유사도 이외의 복잡한 방법 필요
  - 과거에는 Fine-Tuned BERT, T5 등의 모델 사용 — Bi-Encoder VS Cross-Encoder
- LLM을 이용한 Reranking/Reordering
  - LLM을 사용해 관련성을 판단하게 하는 방법 고려

> **도식 예시**: 검색된 순서 A, B, C → LLM/Reranker 모델이 재판단 → 재정렬된 순서 C, A, B

### [실습] 랭체인의 Advanced RAG 기술 (p.51)
- 다양한 모듈을 RAG 파이프라인에 추가하여, 복잡한 RAG 체인 만들기
- RAG 파이프라인의 성능 개선하는 방법 알아보기

## 2. RAG의 성능평가 방법 (p.52)

### RAG: Evaluation (p.52)
수행된 RAG의 성능 평가하기
- Retrieval과 Generation에 대한 평가가 각각 이루어져야 함
  - 더 중요한 것은 Retrieval이나, 이를 정확히 평가하려면 Golden Data 필요
- **RAGAS**: [docs.ragas.io/en/stable](https://docs.ragas.io/en/stable/) — RAG 평가 프레임워크 제공 (평가 LLM, 평가 Embedding 필요)

**평가지표**
1. **Faithfulness**: 답변이 Context에만 근거하여 추출되었는가?
2. **Factual Correctness**: 답변과 Reference가 얼마나 동일한 주장을 하는가?
3. **Context Recall**: Context의 내용이 Reference의 내용을 모두 담는가?
4. **BLEU Score**: 정답과 답변이 얼마나 글자상 일치하는가 (고전적 방법)
5. **Semantic Similarity**: 정답과 답변의 임베딩이 얼마나 일치하는가
6. …

### RAGAS 상세: Claim 기반 평가 방법론 (수업 필기)

**1. Claim(주장)**: 정보/주장의 가장 작은 단위
- LLM을 통해, 정답과 답변을 Claim List로 분할

> 예) "앤트로픽의 코딩 에이전트 클로드 코드 비용이 30% 인하되었다."
> → 1) 클로드는 앤트로픽의 에이전트다. 2) 클로드는 코딩 에이전트다. 3) 클로드 비용이 30% 인하되었다.

**2. Metrics (LLM-as-a-Judge)**

| 지표 | 계산 방법 | 의미 |
|---|---|---|
| **Context Recall** | [검색 결과로부터 유추할 수 있는 Claim 개수] / [정답 Claim 개수] | 낮으면 반드시 올려야 함. 높으면 좋지만 노이즈를 고려해야 함 |
| **Faithfulness** | [검색 결과로부터 유추할 수 있는 Claim 개수] / [답변 Claim 개수] | LLM의 답변이 검색 결과에 근거했는가? 점수가 낮다면 LLM이 Context를 제대로 이해하지 못한 것 |
| **Factual Correctness** | 정답 Claim(레이블) 대비 답변 Claim을 이진 분류 문제의 F1 Score로 계산 | 정답과 답변이 얼마나 동일한 주장을 하는가 |

LLM 기반 평가 프롬프트는 입력 구성에 따라 여러 방식으로 테스트할 수 있습니다.
- (검색결과 - 문제 - 답변 - 정답) 모두 포함해서 테스트
- (문제 - 답변 - 정답)만 포함해서 테스트
- (답변 - 정답)만 포함해서 테스트

**3. 전통적(non-LLM) 평가**
- **ROUGE / BLEU Score**: 정답/답변의 키워드(n-gram) 일치 확인
- **임베딩 유사도**: 정답/답변의 임베딩 유사도 비교

### 현대 RAG의 요구사항 (p.53)
- **많은 정보가 필요한 문제 풀기**
  - Single Retrieval로는 해결하기 어려운 문제
  - Multi-Query, HyDE 등을 조합하여 검색 다각화
  - Deep Research 구조
- **특수한 도메인에서 응답하기**
  - RAG 이외의 지식을 추가 학습
- **임베딩 검색의 한계 극복하기**
  - GraphRAG(그래프 기반 RAG) 등의 구조 고려
  - Multimodal RAG (이미지/테이블 등을 잘 처리하는 RAG) 방식으로의 진화

### RAG in 2026 (p.54)
- LLM의 장문맥 처리 능력이 향상됨
  - 문서가 크지 않다면(100k 미만), 작게 분할하는 대신 Full Context를 넣는 것이 더 안정적
  - 문서가 크다면, 큰 청크를 통해 맥락 단절을 방지함
    - Q) 큰 청크는 검색이 잘 안 되는데? → 큰 청크를 효과적으로 검색하기 위한 방법 고려
- Short Context가 중요한 경우: sLLM
  - Long Context를 처리하지 못한다면, 문제를 쪼개서 RAG를 여러 번 수행하는 것도 방법이 될 수 있음 (개별 청크를 Q/A로 처리하고 종합하기)

### RAG in 2026 (2) (p.55)
- Retrieval → Generation의 단일 워크플로우는 검색 결과에 대한 대응이 불가능함
  - 검색 결과 미비 시 추가 검색, 다중 출처 검색 등
- Retrieval을 활용하는 Agent로의 발전
  - Workflow는 사라지지만, Retrieval 자체는 중요하게 남음
  - Context를 효율적으로 구성하는 방법

> **도식**: 사용자 → LLM(에이전트)이 검색어를 지정하여 검색 요청 → 검색 결과를 바탕으로 추가 검색 OR 답변 진행 → 사용자에게 응답

### RAG의 다양한 시나리오 (수업 필기)

실무에서 마주치는 RAG 질의 유형은 다음과 같이 구분해볼 수 있습니다.

1. **전처리의 중요성**
   - 질문: "어제 회의에서 무슨 얘기했어?" (+ 소속, 이름, 메시지 보낸 시간)
   - → `'2026년 8월 2일 AX전략 1팀 주간 Progress Meeting'`
   - 질문에 포함된 시간, 소속 등 문맥 정보를 전처리하여 검색 쿼리에 활용하면 더 정확한 결과를 얻을 수 있습니다.

2. **의미 기반의 시맨틱 검색**
   - 문서: 사내 포크레인 운용 현황, 사내 덤프트럭 운용 현황, 사내 크레인 운용 현황
   - 질문: "현재 우리 중장비 몇 대 있어?"
   - '포크레인', '덤프트럭', '크레인'이 모두 '중장비'라는 상위 개념으로 묶여 의미적으로 검색되어야 합니다.

3. **키워드 기반의 검색**
   - 문서: 엔비디아 H100 요금표, A100 요금표, A40 요금표, H200 요금표
   - 질문: "A100 80시간 임대하는 비용이 얼마야?"
   - 특정 모델명이나 시간과 같은 키워드를 정확하게 매칭하여 정보를 찾아야 하는 경우입니다.

4. **병렬 검색이 효과적인 질문**
   - 질문: "소설, IT, 역사 관련 이번 달 신작 알려줘."
   - → (2026년 8월 소설 신간), (2026년 8월 IT 신간), … 개별 검색
   - 여러 개의 독립적인 정보 요청이 포함된 질문은 각각의 서브 쿼리로 분할하여 병렬로 검색하는 것이 효율적입니다.

5. **여러 번의 연속 검색이 필요한 질문**
   - 질문: "삼성SDS의 클라우드 산업 경쟁사들이 27년에 진행하려고 하는 프로젝트에 대해 알려줘"
   - 단계:
     1. '삼성SDS 클라우드 산업 경쟁사' 검색 → B사, C사
     2. 'B사 2027 프로젝트' 검색 → 성공
   - 하나의 질문을 해결하기 위해 여러 단계의 검색이 연속적으로 필요하며, 이전 검색 결과가 다음 검색의 쿼리로 사용되는 경우입니다.

> 기존의 RAG 파이프라인(질문 → 검색 → 답변)으로는 5번과 같이 여러 번의 연속 검색이 필요한 복잡한 질문을 효율적으로 해결하기 어렵습니다. → **RAG의 고정적 Workflow에서, Agentic RAG로의 변화 추세** (☜ 위 "RAG in 2026 (2)" 절의 논지와 직결됨)

### [실습] RAG 시스템 성능평가하기 (p.56)
- RAG 파이프라인과 성능평가 모듈 연결하기
- RAGAS의 프롬프트를 이해하고, 좋은 성능평가 방법 직접 고려하기
- 성능평가 방법을 개선하고, 효과적인 RAG 파이프라인 구현하기

### Retrieval 4+1개 유형 (p.57)
- **벡터데이터베이스를 이용한 시맨틱 검색** — 의미가 유사한 문서 가져오기
- **키워드 기반 검색** — 키워드가 겹치는 문서 가져오기 (에이전트를 위한 Ripgrep: [github.com/burntsushi/ripgrep](https://github.com/burntsushi/ripgrep))
- **카테고리형 검색** (폴더처럼 검색하기) — '2026년 5월 2일 회의내용 알려줘' → 2026년 5월 2일 폴더 회의내용 가져오기. 위 방법에 비해 훨씬 결정적
- **그래프 기반 검색** — 전체 데이터를 그래프로 만들고 검색하기
- **(+1) 관계형 데이터베이스 검색** — 질문에 대한 SQL 쿼리 만들고 결과 가져오기

### LightRAG (EMNLP 2025) (p.58)
- GraphRAG 관련 공식 Repo 중 가장 대표적 — [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- LLM 기반의 KG(지식 그래프) 생성/검색 지원
- 벡터 검색 + 그래프 검색을 모두 결합한 듀얼 검색
- 질문 유형에 따른 5가지 쿼리 유형 변경 기능
- Neo4j 등의 DB 없이 바로 시작 가능

## 3. LangChain 실용 예제 (p.59)

### LangChain 실용 예제 개요 (p.60)
LangChain의 부가적인 기능을 활용하여 다양한 작업 가능
- SQL 쿼리 생성하고 실행하기
- Tool과 Agent 만들기
  - **Tool**: LLM이 참고할 수 있는 다양한 외부 기능 (함수, API 등)
  - **Agent**: 함수와 같은 툴을 스스로 활용하여 문제를 해결하는 모듈

### SQL in LangChain (p.61)
- `langchain_community`의 `SQLDatabase`를 이용하여 SQL과 LLM 연결
- SQL Query Chain을 이용한 자동 SQL 쿼리 생성 가능
  - 단, 쿼리 실행의 경우 추가적 작업이 필요하며, 안전 문제를 고려해야 함

### [실습] LangChain을 이용한 SQL 데이터베이스 분석 (p.62)
- 주어진 SQL 데이터에 대해, SQL 쿼리를 자동 생성하는 체인 활용하기
- SQL Query를 실행하는 체인 구성하기

### Tools in LLM (p.63)
**Tool (≈ Function)**: LLM이 외부 함수를 이용할 수 있도록 구현된 모듈
- LLM이 함수 실행 포맷의 출력을 생성할 수 있는 능력을 사전에 학습해야 함

**LangChain의 주요 Tool**
- **SerpAPI**: 구글, 네이버, 유튜브 등의 검색 API를 지원하는 서비스
- **Tavily**: AI 기반 검색 API
- **PythonREPLTool()**: 파이썬 코드를 실행하는 기능 — ChatGPT Code Interpreter와 다르게, 로컬에서 실행

### Tools in LLM — 제어와 자동화 (p.64)
LLM이 제어를 할 수 있을까?
- 제어는 할 수 없지만, 제어 명령은 내릴 수 있다
- Activation을 자동화하는 작업을 연결
- 예) Anthropic의 Computer Use Agent, OpenAI Agent Mode

> **예시**: User: "61번 서버 상태 알려줄래?" → LLM: `Call {Tool: Retrieve_Status, Query: '61'}`

### Tools in LLM — Tool 실행 흐름 (p.65)
Tool이 탑재된 LLM의 판단
- Tool이 필요한 경우, Tool 실행을 요청
- 프로그램상에서 Tool 요청을 파싱하여 실제 실행

Tool 실행 결과: Tool Message
- 툴 실행 결과를 LLM에게 전달
- LLM은 답변 생성

> **예시**: User: "61번 서버 상태 알려줄래?" + LLM: `Call {Tool: Retrieve_Status, Query: '61'}` + Tool: `"504"` → "504 Error로, 게이트웨이 타임아웃으로 추정됩니다."

### LLM Tool Calling — Taker에서 Giver로 (p.66)
외부 Tool과 Text로 소통할 수 있는 인터페이스만 주어지면 양방향 활용 가능
- 파일 시스템 접근, API 호출, Github 이슈 해결 및 커밋, …

> **예시**: User: "(중략) 정리해서 report.txt에 저장해줘." → LLM: `Function Call {Tool: Write_file, Content: '# 최종 정리 내용 …'}` → Tool: `{"Result":"저장 완료!"}` → "리포트 저장이 완료되었습니다!"

### Tools in LangChain (p.67)
Custom 함수의 경우에도, decorator를 통해 tool로 변환 가능
- Argument 정보 및 Description 필요

```python
from langchain_core.tools import tool

@tool
def multiply(x: int, y: int) -> int:
    "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
    return x * y

example = {'x': 10, 'y': 99}
multiply.invoke(example)
```

### Model Context Protocol — MCP (p.68)
Tool Calling을 일반화
- LLM의 Tool 호출 메커니즘을 HTTP/STDIO 통신 기반의 Server/Client 관계로 표준화
  - 크로스 플랫폼
  - 인증/미들웨어 등의 기능 지원
- 다양한 MCP: [smithery.ai](https://smithery.ai/)
- 파이썬으로 MCP 서버/클라이언트를 구현할 때: [FastMCP](https://gofastmcp.com/getting-started/welcome)
- 출처: [medium.com/@halilxibrahim](https://medium.com/@halilxibrahim)

### Agent: 툴을 이용한 복잡한 작업 수행 (p.69)
Tool의 결과물을 얻는 것에서 끝나지 않고, 해당 Tool을 사용해 문제 해결하기
- LLM → 함수 실행 → 함수 결과 → LLM에 다시 전달 → 최종 답변 생성

**Agent**: Planning, Tool Executing, Observation을 포함하여 문제 해결
- Agent의 구성 방식에 따라, 다양한 범위의 작업 수행 가능

### ReAct Agent 프롬프트 (p.70)
- *ReAct: Synergizing Reasoning and Acting in Language Models* — [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Action을 수행할 수 있는 Agent의 프롬프트 방식
  - 생각 → 행동 → 관찰을 통해 다음 행동을 계획하고 실행

**예시**

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

### [실습] LangChain과 다양한 툴 연동하기 (p.71)
LangChain의 Agent와 다양한 툴을 활용하기
- Tavily Search ([tavily.com](https://tavily.com/))
- 툴을 직접 만들고, LangChain 문법으로 연결하기
- `langchain 1.0`의 `create_agent()`를 이용한 에이전트 구현

### Claude Code Skills (p.72)
- 2025년 10월 Anthropic에서 발표한 기능
  - Tool 등의 추가 Context가 너무 많아지는 경우, 해당 내용을 불러오는 Overhead가 너무 많아지는 **Context Bloat** 발생
- Context를 **Foldable 구조**로 구조화하여, 필요한 내용만 불러오는 Skill 제안
  - 최초에는 짧은 설명만을 전달하며, 이후 에이전트가 필요한 스킬을 로드하여 컨텍스트에 저장하는 방법
- 참고: Claude의 기본 동작을 정의하는 [Claude System Prompt (공식 릴리즈 노트)](https://platform.claude.com/docs/en/release-notes/system-prompts)도 유사하게, 모델에 전달되는 지침이 지속적으로 버전 관리/공개되는 사례

### Skill 구성요소 (p.73)
- LLM 전달을 위한 이름과 짧은 설명
  - 예) PPTX: 파이썬을 이용한 PPTX 편집 스킬
- Tool / MCP와 다르게, 형식이 정해져 있지 않은 폴더(Folder) 구조
  - 스킬 사용 설명을 포함하는 `SKILL.md`
  - 특정 작업을 위한 프롬프트
  - Tool 역할의 스크립트
  - 예시 문서
- 예시: [smithery.ai/skills/anthropics/pdf](https://smithery.ai/skills/anthropics/pdf)

### SKILL.md (p.74)
개별 스킬의 내용을 저장하는 표준 규격 문서
- **YAML Frontmatter**와 점진적 공개 메커니즘
  - Frontmatter: 이름과 짧은 소개를 사전에 전달
  - 이후, 해당 스킬을 로드(read)하면, 뒷부분의 내용을 컨텍스트로 가져와 활용하는 방식
    - 지침, 예시, 스크립트 사용법, 템플릿, 다른 스킬 링크, 다른 문서 링크, …

**예시 (`runpod-manager` skill의 frontmatter)**
```yaml
---
name: runpod-manager
description: >
  RunPod GPU Pod를 일괄 생성/관리하는 skill.
  사용자가 (1) 수강생용 GPU Pod를 일괄 생성해 달라고 하거나,
  (2) RunPod Pod 목록/상태를 확인하고 싶거나,
  (3) Pod를 일괄 중지/삭제하고 싶을 때, 또는
  Pod 생성 시 루트 비밀번호 자동 설정, SSH 비밀번호 인증 활성화,
  Jupyter allow_hidden 설정을 자동으로 포함하며,
  접속 정보를 CSV로 출력하여 수강생에게 바로 배포할 수 있다.
---
(이후 229줄의 상세 텍스트 — 원문 슬라이드에 생략 표시)
```

### [실습] LangChain Code Interpreter를 이용한 데이터 시각화 (p.75)
- LangChain과 Python Code Generator 연동하기
  - 데이터 시각화 코드 작성과 개선, 실행 기능 연동하기
  - 시각화 결과 평가 및 개선 모듈 연결하기
  - Skill 개념과 실행 과정 이해하기

---

## 부록: 실습(Lab) 목록 한눈에 보기

| # | 실습명 | 페이지 | 관련 절 |
|---|---|---|---|
| 1 | LangChain 기본 구조 | p.14 | Day1-1 |
| 2 | LangChain을 이용한 데이터 분류와 전처리 | p.18 | Day1-2 |
| 3 | LangChain을 이용한 데이터 생성과 처리 | p.19 | Day1-2 |
| 4 | 벡터데이터베이스 기반 RAG 어플리케이션 | p.37 | Day1-3 |
| 5 | PDF 내용 기반 질의응답 어플리케이션 | p.38 | Day1-3 |
| 6 | 랭체인의 Advanced RAG 기술 | p.51 | Day2-1 |
| 7 | RAG 시스템 성능평가하기 | p.56 | Day2-2 |
| 8 | LangChain을 이용한 SQL 데이터베이스 분석 | p.62 | Day2-3 |
| 9 | LangChain과 다양한 툴 연동하기 | p.71 | Day2-3 |
| 10 | LangChain Code Interpreter를 이용한 데이터 시각화 | p.75 | Day2-3 |

> GPT Builder 관련 실습(원문 p.80, p.82)은 해당 챕터 삭제에 따라 목록에서 제외했습니다.

> 이 저장소(RAG2026)의 `[실습]_*.ipynb` 노트북들이 위 실습 목록에 대응합니다.

## 참고 자료 (원문 인용 출처 모음)

- LangChain 공식 사이트: <https://www.langchain.com/>
- LangChain v1 마이그레이션 가이드: <https://docs.langchain.com/oss/python/migrate/langchain-v1>
- LangChain Document Loaders: <https://python.langchain.com/docs/integrations/document_loaders/>
- Docling: <https://docling-project.github.io/docling/> / Docling-Serve: <https://github.com/docling-project/docling-serve>
- RAGAS: <https://docs.ragas.io/en/stable/>
- LightRAG (EMNLP 2025): <https://github.com/HKUDS/LightRAG>
- Ripgrep: <https://github.com/burntsushi/ripgrep>
- Tavily: <https://tavily.com/>
- Smithery (MCP/Skills 마켓): <https://smithery.ai/>
- HyDE 논문: <https://aclanthology.org/2023.acl-long.99.pdf>
- Lost in the Middle 논문: <https://arxiv.org/pdf/2307.03172>
- Long-Context LLMs Meet RAG 논문: <https://arxiv.org/abs/2410.05983>
- ReAct 논문: <https://arxiv.org/abs/2210.03629>
- OpenAI 모델 목록: <https://platform.openai.com/docs/models>
- Google 모델 목록: <https://ai.google.dev/gemini-api/docs/models?hl=ko>
- HuggingFace: <https://huggingface.co>
- Ollama 모델: <https://ollama.com/models>
- LLM 지능 리더보드 (Artificial Analysis): <https://artificialanalysis.ai/leaderboards/models>
- 임베딩 모델 리더보드 (MTEB, HuggingFace): <https://huggingface.co/spaces/mteb/leaderboard>
- Claude System Prompt (공식 릴리즈 노트): <https://platform.claude.com/docs/en/release-notes/system-prompts>
- Microsoft Harrier 임베딩 (기존 임베딩 모델 파인튜닝): <https://huggingface.co/microsoft/harrier-oss-v1-0.6b>
- FastMCP (파이썬 MCP 프레임워크): <https://gofastmcp.com/getting-started/welcome>
