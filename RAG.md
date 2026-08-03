# 2026년도 삼성 SDS AI 교육: RAG 개발 심화

## 강사 소개
**변형호**
- **학력**
  - 경기과학고등학교 졸업
  - KAIST 전산학부 졸업 (수리과학과 부전공)
  - 서울대학교 컴퓨터공학부 박사 졸업 (2023. 2)
- **경력**
  - 노토랩 대표
  - 삼성전자, 현대자동차, 삼성SDS, SKT, LG AI Research, 현대모비스 등 LLM 개발 강의
  - 국민연금공단 기금운용본부 / 국가보안기술연구소 AI 자문위원
  - 삼성전자 / KT AI 연구소 LLM 트렌드 세미나 진행
  - 두산백과 정보과학/인공지능 분야 집필진
  - LG AI EXAONE 파인 튜닝 온라인/오프라인 강의
- **GitHub**: https://github.com/NotoriousH2

---

## 교육 과정 목표 및 목차

### 과정 목표
1. LLM의 개념을 이해하고, 대표적인 활용 사례를 학습한다.
2. 랭체인을 비롯한 다양한 라이브러리를 학습하고, GPTs, Action 등의 기능을 실습한다.

### 교육 일정
- **Day 1. LangChain 학습**
  1. LangChain 컴포넌트
  2. LangChain으로 자동화 및 챗봇 앱 만들기
  3. 벡터 스토어와 RAG
- **Day 2. RAG 심화-LangChain 개발**
  1. RAG의 다양한 성능 개선 방법
  2. RAG의 성능 평가 방법
  3. LangChain 실용 예제
  4. GPT Builder

---

# Day 1. LangChain 학습

## 1. LangChain 컴포넌트 소개

### LangChain 개요
- **공식 홈페이지**: https://www.langchain.com/
- LLM 어플리케이션 개발에 필요한 다양한 도구 지원
- 높은 추상화(Abstraction) 기반의 쉬운 구현 ("LLM Application Toolbox")
- 자세한 공식 문서 및 다양한 예제 코드 존재 (바이브 코딩 구현도 쉬움)

### LangChain 핵심 모듈
- **모듈식의 구성 요소 연결**:
  - 모델, 프롬프트, 체인
  - 데이터베이스, 메모리
  - 툴 사용과 에이전트

### LangChain 아키텍처 (주요 패키지 구조)
- **Observability**: LangSmith
- **Deployment**: LangServe (Chains as APIs)
- **Cognitive Architectures**: 
  - Templates (Reference Applications)
  - LangChain (Python / JavaScript): Chain / Agents / Advanced Retrieval Strategies
- **Integrations / Components**:
  - LangChain Community (Python / JavaScript):
    - Models I/O: Model, Prompt, Example Selector, Output Parser
    - Retrieval: Retriever, Document Loader, Vector Store, Text Splitter, Embedding Model
    - Agent Tooling: Tools, Toolkits
- **Protocol**:
  - LangChain Core (Python / JavaScript): LCEL (LangChain Expression Language) - Parallelization, Fallbacks, Tracing, Batching, Streaming, Async, Composition
- **스택 종합 지원 기능**: Monitoring, Feedback, Evaluation, Annotation, Playground, Testing, Debugging

### LangChain + LLM Provider
- **LLM Provider별 별도의 라이브러리로 빠른 업데이트**: `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`
- **오픈 모델 및 임베딩 모델 연결**: `langchain_huggingface`, `langchain_ollama` (https://huggingface.co, https://ollama.com/models)

### LangChain 1.0 (2025.10) 개편 사항
- 기존 LangChain 라이브러리의 분리 및 개편
  - `langchain`: LLM Agent 개발을 위한 모듈 위주 (Modern & Future-Focused)
  - `langchain-classic`: 기존의 다양한 도구들 (Stable & Legacy Support)
- 기존 코드가 돌아가지 않는 경우 `langchain_classic`으로 수정하면 대부분 해결됨
- 참고: https://docs.langchain.com/oss/python/migrate/langchain-v1

### Chat Models in LangChain 코드 예시
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

# 실행 예시
gpt5.invoke("안녕? 너는 누구니?")
```

### Runnable과 Invoke
- LangChain의 구성 요소는 **Runnable** 클래스 기반
  - `invoke()`: 동기 실행
  - `ainvoke()`: 비동기 실행
  - `batch()`: 병렬 실행
  - `abatch()`: 병렬 비동기 실행
- `init_chat_model()`을 통한 API 통합 연결: `model_provider`, `model_name` 전달 / `SystemMessage`, `HumanMessage`, `AIMessage` 형태로 메시지 클래스 입력

---

## [실습 1] LangChain 기본 구조
- 기본 프롬프트 구성 방법 이해하기
- 다양한 LLM 모델과 LangChain 연결하기
- 병렬 실행을 위한 `batch`, 순차적 출력을 위한 `stream` 사용하기
- 모델의 프롬프트 템플릿과 LLM 연동하기

---

## 2. LangChain 자동화 및 챗봇 만들기

### LangChain의 Structured Output
- 기존의 LLM은 단순 Message 리스트를 전달하는 방식
- **구조화된 출력**은 별도의 후처리 없이 DB/프롬프트에 바로 연결할 수 있어 LLM 프로그램 구조 고도화에 필수적
- 다양한 파서를 통한 출력 형식 변환 (JSON, Pydantic, DateTime 등)
- LLM의 `with_structured_output()` 기능 활용

### 특수한 Runnables
- 값을 직접 저장하지 않고, Runnable들을 연결하는 역할의 Runnable:
  - `RunnablePassthrough()` : 직전 Runnable의 출력을 그대로 Return
  - `RunnableParallel()` : 1개 이상의 Runnable을 실행하고, 그 결과를 Dict로 Return
  - `.assign()` : Runnable 값을 전달하고, 그 결과를 가져와 결합
- LangChain의 한계점인 Sequential 구성을 보다 다양화 (이후 LangGraph 구조에서 Sequential → Graph 구조로 확장)

---

## [실습 2] LangChain을 이용한 데이터 분류와 전처리
- 분류 프롬프트 구성하기
- LangChain 검색 모듈과 분류 체인 연결하기
- 분류 결과 평가하고 성능 개선하기

---

## [실습 3] LangChain을 이용한 데이터 생성과 처리
- LLM의 구조화된 출력을 사용하여, 다양한 데이터 전처리하기
- 적은 수의 샘플을 사용해, 유사한 데이터 생성하기

---

## 3. Vector Store와 RAG

### LLM 모델의 한계
- 언어의 패턴을 이해하여, 가장 그럴듯한 답변을 제공하는 모델
- 사전 학습(Pretrain)을 통해 지식과 언어적 패턴을 이해하지만 높은 수준이 항상 정확성을 의미하지는 않음
- 최신 데이터/내부 데이터/도메인 특화 데이터 질문 시 **Hallucination (환각)** 위험이 큼

### 검색 증강 생성 (Retrieval Augmented Generation, RAG)
- **LLM과 정보 검색의 결합**: 질의와 관련된 정보를 검색하여 프롬프트에 함께 전달
- LLM의 **In-Context Learning** 능력 기반

### RAG 프롬프트 예시
> 다음의 정보를 참고하여, 질문에 답하세요.
> 
> **정보**: 2025년 5월 13일, 삼성SDS는 웨스틴 조선 서울에서 열린 '금융의 미래를 선도하기 위한 삼성SDS Industry Day' 에서 발전하고 있는 금융 분야의 환경에 대응하기 위해, 금융 업무에 생성형 AI를 적용한 사례를 공유했습니다.
> 
> **질문**: 삼성SDS Industry Day는 언제 열렸나요?

### RAG의 Retrieval (검색)
- 정확한 맥락을 제공하려면 데이터베이스 내용 중 질문과 관련 있는 내용을 알아야 함
- 관련성을 측정 가능한 형태로 수치화하기 위해 텍스트를 벡터로 변환하는 **Embedding/Indexing** 필요

### RAG의 5-Step Process
1. **Indexing**: 데이터베이스 사전 구성
2. **Processing**: 입력 쿼리를 검색이 잘 되도록 전처리
3. **Searching**: 질문과 가장 적합한 데이터 검색하여 나열
4. **Augmenting**: 쿼리와 Searching 결과를 포함하는 프롬프트 구성
5. **Generating**: LLM에 전달해 답변 생성

### Indexing: 데이터 준비하기 (Chunking)
- LLM의 Context는 한정적이므로 전체 문서를 입력하는 것은 불비효율적임
- **Chunking (청킹)**: 전체 텍스트 코퍼스를 글자수/토큰수/챕터 등을 활용하여 분할 저장
- 최적 청킹 기법/청크 크기는 도메인/데이터마다 상이함 (예: OpenAI 기본 RAG 구조 - 800 Chunk / 400 Overlap)
- LangChain의 `RecursiveCharacterTextSplitter()` 등 모듈 활용

### Indexing: 적절한 청크 사이즈 선택하기
- **청크 크기가 작은 경우**: 주변 정보 활용이 어려움
- **청크 크기가 큰 경우**: 불필요한 정보 포함, 임베딩 정확도 감소
- **청크 오버랩(Overlap)**: Context의 의미 보존

### Indexing: 청킹에 대한 견해 변화 (2026 기준)
- **2026 트렌드**: LLM의 장문맥(Long Context) 처리 능력이 대폭 향상됨
- 문서가 크지 않다면 (100k 미만), 작게 분할하는 대신 **Full Context**를 넣는 것이 더 안정적
- 문서가 크다면 큰 청크를 통해 맥락 단절을 방지하고, 큰 청크를 효과적으로 검색하기 위한 방안을 고려

### LangChain Document Loader & Docling
- **Document Loader**: 다양한 형식의 파일(PDF, DOCX, XML)이나 API 서비스(Arxiv, Tavily, Pubmed) 연결 모듈
- **Docling (구조화된 데이터 추출 파서)**:
  - PDF/PPTX/DOCX/HTML 문서를 JSON, Markdown, Figures로 변환
  - 이미지 및 테이블 추출에 강함
  - Docling-Serve를 통한 API 서빙 지원 (https://docling-project.github.io/docling/)

### Processing: 사용자 쿼리 처리하기
- 의도 분류 / 작업 선택 / 쿼리 재구성
- 어떤 DB로 검색할 문제인지 사전 파악
- **짧은 쿼리**: 맥락 이해 정보가 부족하므로 이전 대화나 추가 정보 고려하여 Contextualize
- **긴 쿼리**: 불필요한 정보 제거 또는 요약

### Searching: 쿼리와 적합한 데이터 검색하기
- **Dense Retrieval (Semantic 검색)**: 트랜스포머 모델을 활용해 텍스트를 Dense 벡터로 변환, 의미적 유사성 탐색
- **Sparse Retrieval (Lexical 검색)**: BM25, TF-IDF, SPLADE 기법 활용, 키워드 일치 여부에 높은 가중치

### Vector Database (임베딩 저장 공간)
- 비정형 데이터를 임베딩으로 변환하여 저장 및 검색
- **작동 원리**: 질문의 임베딩과 저장된 청크 임베딩의 유사도를 계산하여 Top-K Chunks 반환
- **주요 Vector DB**:
  - 전용 DB: Pinecone(유료 클라우드), Milvus, Qdrant, Chroma, Weaviate
  - 기존 DB의 벡터 검색 지원: ElasticSearch, OpenSearch (벡터 수가 적고 기존 DB 운영 시 권장)
- **Metric Types**: Euclidean Distance (L2), Cosine Distance, MMR (Maximum Marginal Relevance Search)

---

## [실습 4] 벡터 데이터베이스 기반 RAG 어플리케이션
- `WebBaseLoader()`를 통한 웹 페이지 내용 기반 RAG
- ChromaDB를 이용하여 뉴스 기사 청크 저장 및 검색

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

def format_docs(docs):
    return "\n--\n".join([doc.page_content + '\nURL: ' + doc.metadata['source'] for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## [실습 5] PDF 내용 기반 질의응답 어플리케이션
- PDF 파일의 내용을 기반으로 연속적 질의응답 수행
- 요약 알고리즘의 stuff, map-reduce, refine 과정 수행

```python
reduce_prompt = ChatPromptTemplate([
    ('system', '''당신은 인공지능과 거대 언어 모델의 전문가입니다.
LLM 프로그래밍에 대한 요약문의 리스트가 주어집니다.
이를 읽고, 전체 주제를 포함하는 최종 요약을 작성하세요.
요약은 5개의 문단과 문단별 4-8개의 문장으로 작성하세요.'''),
    ('user', '''{text}
Summary:''')
])

reduce_chain = reduce_prompt | llm | StrOutputParser()
summary = reduce_chain.invoke('\n--\n'.join(raw_summaries))
```

---

# Day 2. RAG 심화 - LangChain 개발

## 1. RAG의 다양한 성능 개선 방법

### RAG 파이프라인 요소와 성능 영향
- Chunking (Size / Top K), Searching (Semantic / Lexical), 순서, 비중, 다중 검색 등 모든 요소가 성능에 직접 영향

### Small2Big Chunking / Sliding Window
- **Small2Big Chunking**: 검색 정확도를 위해 청크 사이즈를 작게 구성하여 검색하고, 실제 Augmenting 단계에서는 연결된 큰 청크(Big Chunk)를 프롬프트에 전달 (LangChain: `ParentDocumentRetriever`)
- **Sliding Window**: Small Chunk의 위치 기준으로 앞뒤 문맥 포함

### Contextual Retrieval
- 청킹 시 앞부분 단절을 완화하기 위해 전체 맥락 정보(헤더, 문서 정보)를 청크 앞부분에 추가
- Context Caching을 활용하여 비용 절감 가능 (Claude 보고서 등에서 Hybrid 검색 성능 대폭 향상 확인)

### Advanced Processing (Query 변환)
- **Multi-Query Retrieval**: 쿼리를 여러 개 버전으로 세분화하여 각각 검색 수행
- **Query Reconstruction**: 대화 맥락을 고려하여 사용자 쿼리를 검색에 유용하게 재구성
- **HyDE (Hypothetical Document Embedding)**: LLM을 이용해 질의에 대한 가상의 답변/문서를 먼저 생성한 후, 해당 가상 문서를 기반으로 DB를 검색하여 정확도 향상

### Advanced Searching & Augmenting
- **Hybrid Search**: Lexical과 Semantic 검색 결과를 모두 고려 (Reciprocal Rank Fusion 사용)
- **Metadata Filtering**: 전처리 단계에서 카테고리, 페이지, 날짜, 작성자 등의 메타데이터 부여 및 필터링
- **Long Context Model의 시대**: Context Window 증대 (Gemini 3.1 Pro Preview 1M, GPT-5.4 1.05M, Claude Opus 4.6 1M 등)
- **Lost-in-the-middle Error**: 긴 Context 전달 시 중간에 위치한 중요 정답 정보를 놓치는 현상 완화 필요 → Context Optimization (중요 정보 위치 조절, 불필요 청크 제거)
- **Reranking**: Bi-Encoder/Cross-Encoder 모델 또는 LLM을 이용하여 질문과 검색 문서 간의 관련도를 평가해 재정렬 및 필터링

---

## [실습 6] 랭체인의 Advanced RAG 기술
- 다양한 모듈을 RAG 파이프라인에 추가하여 복잡한 RAG 체인 구축
- RAG 파이프라인의 성능 개선 기법 실습

---

## 2. RAG의 성능 평가 방법 (Evaluation)

### RAG Evaluation 개념
- Retrieval과 Generation에 대한 개별 및 종합 평가 필요
- 정확한 평가를 위해서는 **Golden Data** 구축이 중요함

### RAGAS 프롬프트 및 평가 항목 (https://docs.ragas.io/)
1. **Faithfulness**: 답변이 주어진 Context에만 근거하여 작성되었는가?
2. **Factual Correctness**: 답변과 Reference가 얼마나 동일한 주장을 하는가?
3. **Context Recall**: Context가 Reference의 내용을 모두 담고 있는가?
4. **BLEU Score**: 정답과 답변의 텍스트 표면적 일치도 (고전적 기법)
5. **Semantic Similarity**: 정답과 답변의 임베딩 유사도

### 현대 RAG / RAG in 2026 트렌드
- **Deep Research / Agentic RAG**: Single Retrieval 한계를 극복하기 위해 Multi-Query, HyDE 및 검색 실패 시 재검색, 다중 출처 검색을 수행하는 Agent 구조로 발전
- **Retrieval 4+1 유형**:
  1. 벡터 DB 시맨틱 검색
  2. 키워드 기반 검색 (Ripgrep 등)
  3. 카테고리형 검색 (폴더/날짜 기반)
  4. 그래프 기반 검색 (GraphRAG / LightRAG - EMNLP 2025)
  5. (+1) 관계형 DB 검색 (Text-to-SQL)
- **LightRAG**: LLM 기반 Knowledge Graph 생성 및 벡터+그래프 듀얼 검색 지원 (Neo4j 없이 구동 가능)

---

## [실습 7] RAG 시스템 성능 평가하기
- RAG 파이프라인과 성능 평가 모듈 연결
- RAGAS 프롬프트를 이해하고 효과적인 평가 메트릭 적용

---

## 3. LangChain 실용 예제

### SQL in LangChain
- `langchain_community.utilities.SQLDatabase` 사용
- `QuerySQLDataBaseTool`을 이용한 자동 SQL 쿼리 생성 및 실행 (보안 및 실행 안전성 고려 필요)

```python
answer_prompt = PromptTemplate(template="""
당신은 매우 활발하고 유머러스한 도서관 사서 AI입니다.
모든 대화는 책 속 유명한 인물의 말투를 그대로 과장되게 따라하세요.
맨 뒤에, 괄호를 통해 누구인지 알려주세요. (000 톤으로)
질문과 SQL 쿼리, 쿼리의 실행 결과가 주어집니다.
해당 정보를 바탕으로 질문에 대한 답변을 생성하세요.

Question: {question}
SQL Query: {query}
SQL Result: {result}
""")

answer_chain = answer_prompt | llm | StrOutputParser()
chain = RunnableParallel(question=RunnablePassthrough()).assign(query=query_chain | parse_sql)
```

### Tools & Agents in LangChain
- **Tool**: LLM이 외부 함수나 API를 실행할 수 있도록 연결한 모듈 (`SerpAPI`, `Tavily`, `PythonREPLTool`, Custom Decorator `@tool`)
- **Agent**: Planning, Tool Executing, Observation 단계를 거쳐 스스로 외부 도구를 활용해 문제를 해결하는 모듈
- **ReAct Framework**: Thought → Action → Observation 흐름으로 문제 해결
- **Model Context Protocol (MCP)**: Tool Calling 기법을 HTTP/STDIO 통신 기반의 Client/Server 구조로 표준화한 크로스 플랫폼 규격 (Database, Web API, GitHub, Slack, Gmail, Local Filesystem 연결)

### Claude Code Skills / Skill 구조
- **Context Bloat 방지**: 많은 컨텍스트를 Foldable 구조로 조직화하여 필요 시 로드하는 메커니즘
- **SKILL.md**: YAML Frontmatter(이름, 짧은 소개)와 상세 지침, 프롬프트, 스크립트를 포함하는 규격 문서

---

## [실습 8] LangChain과 다양한 툴 연동 / SQL DB 분석
- Tavily Search 및 Custom Tool 구현
- `create_agent()`를 이용한 에이전트 구축
- SQL 쿼리 자동 생성 및 대화형 질의응답 체인 구축

---

## [실습 9] LangChain Code Interpreter를 이용한 데이터 시각화
- Python Code Generator 연동 및 시각화 코드 작성/개선
- 시각화 결과 평가 및 Skill 실행 구조 이해

---

## 4. GPT Builder & Actions

### GPTs 및 GPT Builder
- OpenAI ChatPlatform 상에서 Custom GPT 구축 (`Code Interpreter`, `Retrieval`, `Image Generation` 내장)
- GPT Builder와의 대화를 통해 지침(Instructions), 프롬프트, 구성을 자동으로 정의

### GPTs Actions
- 외부 REST API 및 서비스 연동을 위한 OpenAPI Schema (`3.1.0`) 지정
- OAuth 인증 및 Zapier AI Actions 등과 연동하여 확장성 제공

---

## [실습 10] GPTs Builder로 GPTs 구현 및 Action 연동
- GPT Builder를 통한 커스텀 GPT 제작
- Action Schema 문법을 활용한 외부 API 액션 연동
