# Lab 5. PDF 내용 기반 질의응답 어플리케이션

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"Indexing: 데이터 준비하기"(p.26)~"Searching"(p.32~36)** 및 **"RAG의 5-step Process"(p.25)** 이론을, `[실습]_5_PDF_내용_기반_질의응답_어플리케이션.ipynb`의 실제 실행 결과와 결합하여 재구성한 것입니다. README 부록 "실습(Lab) 목록"의 **Lab #5 "PDF 내용 기반 질의응답 어플리케이션"(p.38, Day1-3)**에 해당합니다.

> ℹ️ **실행 환경 참고**: 이 노트북은 상단에 Colab 배지와 `google.colab.drive.mount()` 코드가 있는 것에서 알 수 있듯, **Google Colab 환경에서 원본 그대로 실행된 결과**가 기록되어 있습니다(Lab 6·7·8 교재처럼 Windows 로컬에서 재실행한 기록이 아닙니다). 이 교재에서는 노트북에 실제로 남아 있는 Colab 실행 결과를 그대로 인용합니다.

## 학습 목표

- PDF 문서를 페이지 단위로 불러오고, **토큰 기반 청킹**으로 재구성하여 벡터스토어에 저장하는 전체 Indexing 과정을 이해한다.
- Chroma 벡터스토어와 Retriever를 구성하여 **Top-K 검색 기반 RAG 질의응답 체인**을 직접 만들어본다.
- Top-K 기반 RAG의 한계(전체 문서를 조망해야 하는 질문에는 취약함)를 실제로 확인하고, 이를 보완하는 **3가지 문서 요약 전략(Stuff / Map-Reduce / Refine)**의 원리와 트레이드오프를 실습으로 체득한다.
- 이론(Indexing/Chunking/Long Context/Lost-in-the-middle)과 실습 결과를 연결하여, "언제 어떤 방법을 써야 하는지"에 대한 판단 기준을 세운다.

## Part 0. 개요와 노트북 구조

**노트북 44개 셀의 흐름**:
1. 라이브러리 설치, Google Drive 마운트, LLM 준비
2. **Indexing**: PDF 로드(`PyMuPDFLoader`) → 전체 문서 병합 → 토큰 기반 청킹 → Chroma 벡터스토어 구성
3. Retriever로 검색 테스트, RAG 질의응답 체인 구성 및 실행
4. Top-K RAG의 한계 논의 → **요약(Summarization)**: Stuff / Map-Reduce / Refine 3가지 방식 구현·비교
5. [실습 과제] 임의의 PDF를 다운로드하여 동일한 방식으로 요약하기 (미완성 상태로 남겨진 실습)

> 💡 이 노트북에서 RAG의 대상 문서는 다름 아닌 **`교재_0803.pdf`(이 README.md의 원본 PDF) 자기 자신**입니다 — "이 강의 교재에 대해 강의 교재로 질의응답한다"는 재귀적인 구조로 되어 있어, README 이론과 실습 결과를 대조하기에 특히 적합합니다.

---

## Part 1. Indexing: PDF 로드와 청킹

### 이론: RAG의 5-step Process — Indexing (p.25~26)

> 1. **Indexing** — 데이터베이스 사전 구성
>
> - LLM의 Context는 한정적이므로, 전체 문서를 입력하는 것은 효과적이지 않음
>   - 여러 문서가 포함된 경우에도, 필요한 부분만 선택적으로 검색할 수 있도록 분리
>   - 예) 페이지/문단 단위로 나누어 저장
> - **Chunking(청킹)**: 전체 텍스트 코퍼스를 분할하는 작업
>   - 주로 글자수/토큰수/챕터 등을 활용하여 분할
>   - OpenAI의 기본 RAG 구조: 800 Chunk / 400 Overlap

### 실습: 라이브러리 설치와 LLM 준비

```python
!pip install openai langchain langchain-openai langchain-community chromadb tiktoken langchain_chroma pymupdf -q
```

**실행 결과** (Colab, 패키지 설치 로그):
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-colab 1.0.0 requires requests==2.32.4, but you have requests 2.34.2 which is incompatible.
google-adk 2.4.0 requires opentelemetry-api<=1.42.1,>=1.39, but you have opentelemetry-api 1.44.0 which is incompatible.
google-adk 2.4.0 requires opentelemetry-sdk<=1.42.1,>=1.39, but you have opentelemetry-sdk 1.44.0 which is incompatible.
```

> ℹ️ 이 경고는 설치 자체의 실패가 아니라, Colab 기본 이미지에 이미 설치된 `google-colab`/`google-adk` 패키지가 요구하는 버전과 이번에 새로 설치한 `requests`/`opentelemetry-*` 버전이 살짝 어긋난다는 pip의 알림입니다. 이번 실습(랭체인 관련 패키지)의 동작에는 영향이 없습니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

**실행 결과**: `Mounted at /content/drive`

```python
import os
from dotenv import load_dotenv
load_dotenv(override=True)

if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')
```

**실행 결과**: `OpenAI API 키 확인`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.6", reasoning_effort='low')
```

### 실습: PDF 불러오기 — `PyMuPDFLoader`

### 이론: LangChain Document Loader (p.29)

> 다양한 형식의 파일이나, API 서비스를 연결하여 데이터를 불러오는 구조
> - PDF, DOCX, XML 등의 파일 로더
> - Arxiv, Tavily, Pubmed 등의 API 로더

```python
# 현재 위치의 모든 pdf 불러오기
from glob import glob
import os

pdfs = glob(os.path.join('./', '*.pdf'))
pdfs
```

**실행 결과**: `['./교재_0803.pdf']`

```python
from langchain_community.document_loaders import PyMuPDFLoader

documents = []

for path_material in pdfs:
    print(path_material)
    loader = PyMuPDFLoader(path_material)
    # 페이지별로 저장
    pages = loader.load()
    documents.extend(pages)
    print("# Number of pages:", len(pages))

print("# Total pages:", len(documents))
```

**실행 결과**:
```
/tmp/ipykernel_1824/4215178708.py:1: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import PyMuPDFLoader

./교재_0803.pdf
# Number of pages: 82
# Total pages: 82
```

> 💡 **이론 연결 — LangChain 1.0 (p.11)**: 이 Deprecation 경고는 README의 "LangChain 1.0 (2025.10)" 개편 내용과 정확히 같은 맥락입니다. `langchain_community`는 유지보수가 종료되는 방향이며, 공식적으로는 개별 통합 패키지(예: 향후의 `langchain-pymupdf` 유사 패키지)로 이전할 것을 권장합니다. 다만 실습 시점에는 여전히 정상 동작합니다.
>
> **82페이지 = `PyMuPDFLoader`가 로드한 페이지 수**입니다. 이는 다름 아닌 `교재_0803.pdf`(이 README.md의 원본) 전체 페이지 수와 일치합니다 — README 서두의 "총 82페이지" 설명과 맞아떨어집니다.

### 실습: 페이지 단위로 저장된 데이터 확인

```python
for i in range(10,15):
    print(documents[i].page_content)
    print('----------')
```

**실행 결과**:
```
LangChain 1.0 (2025.10)
기존LangChain 라이브러리의분리및개편
- langchain : LLM Agent 개발을위한모듈위주
- langchain-classic: 기존의다양한도구들
기존코드가돌아가지않는경우
- langchain_classic으로수정하면대부분해결됨
- https://docs.langchain.com/oss/python/migrate/langchain-v1
11
----------
Chat Models in LangChain
12
----------
Runnable과Invoke
LangChain의구성요소는Runnable 클래스
- Runnable 계열의클래스는invoke()를통해동기실행, ainvoke()로비동기실행
- batch()로병렬실행, abatch()로병렬비동기실행
Init_chat_model()을통한API 통합연결
- Model_provider, model_name 전달
- SystemMessage, HumanMessage, AIMessage 형태로메시지클래스입력
13
----------
[실습] LangChain 기본구조
기본프롬프트구성방법이해하기
다양한LLM 모델과LangChain 연결하기
병렬실행을위한batch, 순차적출력을위한stream 사용하기
모델의프롬프트템플릿과LLM 연동하기
14
----------
LangChain 자동화및챗봇만들기
15
----------
```

> ⚠️ **PDF 추출 과정에서 관찰되는 특징**: 한글 문장에서 **띄어쓰기가 모두 사라진 것**("기존LangChain 라이브러리의분리및개편")을 볼 수 있습니다. 이는 `PyMuPDFLoader`가 PDF의 텍스트 레이어를 위치 기반으로 추출하는 과정에서, 슬라이드 원본의 폰트/자간 렌더링 방식에 따라 단어 사이 공백 문자가 유실되는 흔한 현상입니다. 이후 벡터 임베딩 자체에는 큰 영향을 주지 않지만(임베딩 모델이 어느 정도 이런 노이즈에 강건함), **사람이 읽는 표시용 텍스트**로 쓰려면 별도 정제(예: `re.sub`로 CJK 문자 사이에 공백 삽입)가 필요할 수 있습니다. (Lab 6에서는 `preprocess()` 함수로 특수문자/중복 공백을 정리하는 전처리를 실제로 수행합니다.)

### 실습: 전체 문서를 하나로 합치기

```python
from langchain_core.documents import Document
# Document 클래스 만들기

corpus = Document(page_content='', metadata={'source': ', '.join(pdfs)})
for page in documents:
    corpus.page_content += page.page_content + '\n'

corpus.page_content = corpus.page_content.replace('\n\n','\n')
len(corpus.page_content)
```

**실행 결과**: `14592` (전체 82페이지 텍스트를 합친 글자 수)

### 실습: 토큰 단위 청킹

### 이론: 적절한 청크 사이즈 선택하기 (p.27) + 청킹에 대한 견해 변화 (p.28)

> - 청크 크기가 작은 경우 — 주변 정보 활용 어려움
> - 청크 크기가 큰 경우 — 불필요한 정보 포함, 임베딩의 정확도 감소
> - 청크 오버랩 — Context의 의미 보존
>
> **2026: LLM의 장문맥 처리 능력이 향상됨**
> - 문서가 크지 않다면(100k 미만), 작게 분할하는 대신 Full Context를 넣는 것이 더 안정적

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = token_splitter.split_documents([corpus])
print(len(chunks))
```

**실행 결과**: `11`

> 💡 **이론 연결**: 전체 코퍼스가 14,592자에 불과해 README p.28의 "100k 미만이면 Full Context가 더 안정적"이라는 기준에 훨씬 못 미칩니다. 즉 이 문서는 이론상 **청킹 없이 통째로 LLM에 넣어도 무리가 없는 크기**입니다. 그럼에도 이 실습에서는 학습 목적으로 `chunk_size=1000, chunk_overlap=200`(오버랩 20%)의 토큰 기반 청킹을 적용하여 11개 청크로 나눕니다. `.from_tiktoken_encoder()`를 사용하면 **글자 수가 아니라 실제 토큰 수 기준**으로 분할되므로, LLM의 Context 제한과 더 정확하게 맞아떨어집니다.

---

## Part 2. Chroma 벡터스토어 구성과 Retrieval

### 이론: Searching — 벡터데이터베이스 (p.32~36)

> - Embedding 기반의 Semantic 검색(DENSE): 정확하게 일치하지 않아도 유사한 의미를 탐색
> - 벡터데이터베이스 = 임베딩 저장 공간: 질문의 임베딩과 청크들의 임베딩 유사도를 계산해 Top K Chunks를 Return
> - 대표적 Vector Database: Pinecone(클라우드), **Milvus, Qdrant, Chroma, Weaviate**(Online/Self Host)

이 실습은 위 목록 중 **Chroma**를 사용합니다(Lab 4·6에서는 Qdrant를 사용 — README가 소개하는 여러 Self-host 벡터DB 중 실습마다 다른 것을 선택해 경험해보도록 구성되어 있습니다).

### 실습: Chroma에 저장하고 Retriever 만들기

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
Chroma().delete_collection()
db = Chroma.from_documents(chunks, embeddings)


retriever = db.as_retriever(search_kwargs={"k": 5})
# Top K search 옵션 정하기


# filter 옵션을 통해 특정 메타데이터를 가진 벡터만 검색 가능
# Ex) author가 Hyungho Byun인 Document만 검색
# retriever = db.as_retriever(search_kwargs={"k": 5,"filter":{'author':'Hyungho Byun'}})
```

> 💡 주석으로 남겨진 `filter` 옵션 예시는 README "Searching: 벡터DB 활용 예시"(p.34)의 메타데이터 활용과, Lab 6 이론 "Advanced RAG (Searching) — Metadata Filtering"(p.46, "메타데이터를 통해 관련 정보만 선택 — 카테고리, 페이지, 날짜, 작성자 등")을 미리 보여주는 대목입니다.

### 실습: 검색 테스트

```python
# Query 검색
unique_docs = retriever.invoke("LangChain의 장점은?")

unique_docs
```

**실행 결과** (Top-5 청크, `text-embedding-3-large` 임베딩 기준 유사도 상위):
```python
[Document(id='02c9287c-1a73-404f-aa70-646b486efe5d', metadata={'source': './교재_0803.pdf'}, page_content='- 질문이입력되면, 질문의임베딩과청크들의임베딩유사도계산\n- Top K Chunks를Return\n청크검색후, Task 에따른활용\nRAG : LLM 프롬프트에추가하여답변생성\nRecommendation : 해당상품혹은문서전달\n34\nSearching: 대표적Vector Database\n... (p.34~43 구간의 검색·성능개선 이론 슬라이드 텍스트) ...'),
 Document(id='93e1ed31-59fc-4d47-a37c-2f32c857018b', metadata={'source': './교재_0803.pdf'}, page_content='2026년도삼성SDS AI 교육\nRAG 개발심화\n변형호\n[학력]\n... (p.1~13 구간: 강사 소개, 과정 개요, LangChain 컴포넌트 이론) ...'),
 Document(id='e6790637-38fe-4569-911c-7f2a0f3b5f94', metadata={'source': './교재_0803.pdf'}, page_content='53\nRAG in 2026\n... (p.53~70 구간: RAG in 2026, Retrieval 4+1유형, LangChain 실용예제, ReAct) ...'),
 Document(id='50cc0193-1b7d-4c71-b6e1-afcd3dbb2031', metadata={'source': './교재_0803.pdf'}, page_content='찾는것이중요합니다.\nIndexing: 청킹에대한견해변화\n... (p.27~36 구간: 청킹, Document Loader, Docling, Processing, Searching 이론) ...'),
 Document(id='f42e1f73-75b7-4a12-95e2-8147edd5a978', metadata={'source': './교재_0803.pdf'}, page_content='－Agent: 함수와같은툴을스스로활용하여문제를해결하는모듈\n60\nSQL in LangChain\n... (p.60~70 구간: SQL, Tool, MCP, Agent, ReAct 이론) ...')]
```

> ⚠️ **주목할 점**: "LangChain의 장점은?"이라는 질문에 대해 정작 **1번 청크(LangChain 소개, p.7~8)**는 상위 5개에 포함되지 않았고, 대신 검색·RAG 성능개선·Tool/Agent 관련 청크들이 검색되었습니다. 이는 (1) 청크 크기가 1000토큰으로 비교적 커서 여러 주제가 한 청크에 뒤섞여 있고, (2) "장점"이라는 표현이 벡터 임베딩 상 "RAG/검색/구조" 관련 슬라이드들과도 느슨하게 유사하게 잡혔기 때문으로 보입니다. 실제로 다음 Part의 RAG 답변은 그럼에도 합리적인 답을 생성했는데, 이는 상위 5개 청크 안에 LangChain 소개 문구("모듈식의 구성 요소 연결", "높은 추상화 기반의 쉬운 구현" 등)가 부분적으로 흩어져 포함되어 있었기 때문입니다. → **이것이 바로 이 실습 후반부에서 다룰 "Top-K 검색의 한계"의 실제 사례**입니다.

---

## Part 3. RAG 질의응답 체인 구성

### 이론: RAG 프롬프트 예시 (p.23) + RAG의 Retrieval: 검색 (p.24)

> 정보와 질문을 구분하여 제공 — 정확한 정보의 검색이 무엇보다 중요

### 실습: [실습] PDF 질의응답 체인 만들기

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 주어진 문맥을 바탕으로 질문에 답변하는 친절한 AI 어시스턴트입니다.
    주어진 정보를 벗어나서 답변하지 마세요.
    모르는 내용은 모른다고 솔직하게 답변하세요.\n\n{context}"""),
    ("human", "{question}")
])
```

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # retriever가 context를 제공하고, question은 그대로 전달
    | prompt # prompt 템플릿에 context와 question을 전달
    | llm # LLM을 호출하여 답변 생성
    | StrOutputParser() # LLM의 출력을 문자열로 파싱
)
```

> ℹ️ Part 1~2에서 다룬 4단계(Loading → Splitting → Embedding+Storing → Retrieving)에 이어, 이 셀에서 **Augmenting(프롬프트 구성)과 Generating(LLM 호출)**까지 연결되어 README의 "RAG의 5-step Process"(p.25)가 코드 3줄로 완성됩니다. `{"context": retriever, ...}` 부분에서 `retriever`가 `Runnable`이므로 곧바로 딕셔너리 값으로 연결할 수 있다는 점도 눈여겨볼 부분입니다.

```python
# 테스트
rag_chain.invoke("LangChain의 장점은?")
```

**실행 결과**:
```
'LangChain의 주요 장점은 다음과 같습니다.\n\n- **높은 추상화**를 제공해 LLM 애플리케이션을 쉽게 구현할 수 있음\n- 모델, 프롬프트, 체인, 데이터베이스, 메모리 등을 **모듈식으로 연결**할 수 있음\n- **툴 사용과 에이전트 개발**을 지원함\n- 다양한 LLM 제공자와 오픈 모델을 별도 라이브러리로 연결할 수 있음\n- 공식 문서와 예제 코드가 풍부해 **바이브 코딩으로도 구현하기 쉬움**\n\n즉, LangChain은 LLM 애플리케이션 개발에 필요한 기능을 모아 놓은 **“LLM Application Toolbox”**라고 할 수 있습니다.'
```

Part 2에서 확인한 것처럼 상위 5개 검색 청크에 "LangChain 소개" 청크가 온전히 포함되지는 않았지만, LLM이 흩어진 단서(모듈식 구성, 높은 추상화, Tool/Agent 등)를 잘 종합하여 **README의 "LangChain (p.7)"·"LangChain 구조 (p.8)" 내용과 실질적으로 일치하는 답변**을 생성했습니다.

---

## Part 4. Top-K RAG의 한계와 요약(Summarization)

### 이론적 문제 제기 (노트북 원문)

> Top-K 기반의 RAG도 문서의 내용을 바탕으로 답변하지만, 특정 문제에 대해서는 단일 검색이 아닌 전체 문서를 확인해야 하는 경우가 존재합니다. Chunking을 활용하여, PDF 파일을 요약해 보겠습니다.

### 이론 연결: 왜 Top-K 검색만으로는 부족한가

- **Advanced RAG (Augmenting) — Long Context (p.47)**: "전보다 많은 데이터를 처리 — RAG에 넣을 수 있는 Context도 증가(Larger K, Bigger Chunk)". 하지만 K를 무한정 늘릴 수는 없고, 늘리더라도:
- **Advanced RAG (Augmenting) — Lost-in-the-middle Error (p.48)**: "정답이 포함되어 있음에도 불구하고, 답변의 성능이 떨어지는 현상 — 앞뒤의 Context가 노이즈로 작용"
- **Indexing: 청킹에 대한 견해 변화 (p.28)**: "문서가 크지 않다면(100k 미만), 작게 분할하는 대신 Full Context를 넣는 것이 더 안정적"

즉, **"전체 문서를 요약해서 파악해야 하는 질문"**(예: "이 문서 전체를 요약해줘", "이 교재의 핵심 주제 10가지는?")에는 Top-K 검색이 근본적으로 부적합합니다 — 검색은 "관련 있는 일부"만 가져오는 방법이기 때문입니다. 이 문제에 대한 전통적인 해법이 바로 **Stuff / Map-Reduce / Refine** 3가지 요약 전략입니다(LangChain 초기 버전부터 제공되던 `load_summarize_chain`의 3가지 모드로, README에는 별도 이론 챕터가 없지만 실무에서 매우 자주 쓰이는 패턴이라 아래에 정리합니다).

### 참고: 3가지 요약 전략 한눈에 비교

| 전략 | 원리 | 장점 | 단점 |
|---|---|---|---|
| **Stuff** | 전체 문서를 한 번에 프롬프트에 넣고 요약 | 구현이 가장 간단, 문맥 손실 없음 | 문서가 LLM의 Context Length보다 크면 실행 불가 |
| **Map-Reduce** | (Map) 각 청크를 병렬로 개별 요약 → (Reduce) 개별 요약들을 통합 | 매우 긴 문서도 처리 가능, 병렬 처리로 속도 향상 | 청크 간 맥락 손실 가능, API 호출 횟수 증가 |
| **Refine** | 첫 청크로 초기 요약 생성 → 다음 청크를 볼 때마다 요약을 순차적으로 개선 | 문서의 맥락을 순차적으로 유지 | 순차 처리라 느림, 뒷부분 정보가 앞부분 요약을 희석시킬 수 있음 |

---

### Part 4.1 Stuff: 전체 문서를 다 넣고 요약하기

> 가장 간단한 요약 방법입니다. 문서의 길이가 Context 길이보다 큰 경우에는 실행이 어렵습니다.

이번 실습 문서(14,592자, 약 11개 토큰 청크)는 최신 LLM의 Context 길이에 비해 매우 작으므로 Stuff 방식이 문제없이 동작합니다.

```python
from langchain_core.prompts import ChatPromptTemplate

stuff_prompt = ChatPromptTemplate([
    ("system", """당신은 전문적인 문서 요약 전문가입니다.
주어진 문서를 읽고 핵심 내용을 체계적으로 요약해주세요.

요약 가이드라인:
1. 문서의 주요 주제와 목적을 먼저 파악하세요
2. 핵심 내용을 구조화하여 정리하세요
3. 중요한 수치나 데이터는 포함하세요
4. 전문 용어는 그대로 사용하되, 맥락을 명확히 하세요"""),
    ("human", """다음 문서를 요약해주세요.

---
{text}
---

위 문서의 핵심 내용을 체계적으로 요약해주세요.""")
])

# Stuff 체인 구성
stuff_chain = stuff_prompt | llm | StrOutputParser()
```

```python
import time
# Stuff 방식 실행
print("Stuff 방식으로 요약 중...")
start_time = time.time()

stuff_summary = stuff_chain.invoke({"text": corpus.page_content})

elapsed_time = time.time() - start_time
print(f"완료! (소요 시간: {elapsed_time:.2f}초)\n")
print("="*60)
print("[Stuff 요약 결과]")
print("="*60)
print(stuff_summary)
```

**실행 결과** (소요 시간 40.07초, 전체 82페이지를 한 번에 요약):
```
Stuff 방식으로 요약 중...
완료! (소요 시간: 40.07초)

============================================================
[Stuff 요약 결과]
============================================================
# 문서 요약: 2026 삼성SDS AI 교육 ― RAG 개발 심화

## 1. 교육 개요와 목적

본 문서는 **LangChain을 활용한 LLM 애플리케이션 및 고급 RAG(Retrieval Augmented Generation) 시스템 개발**을 다루는 2일 과정의 교육자료다.

### 과정 목표
- LLM의 기본 개념과 대표 활용 사례 이해
- LangChain의 주요 컴포넌트와 개발 방식 습득
- 벡터 데이터베이스 기반 RAG 구현
- RAG 검색·생성 성능 개선 및 평가
- Tool, Agent, SQL, MCP 등 실용 기능 활용
- GPT Builder와 Action을 이용한 커스텀 GPT 개발

### 과정 구성
- **Day 1:** LangChain 컴포넌트, 자동화·챗봇, 벡터 스토어와 RAG
- **Day 2:** Advanced RAG, RAG 평가, LangChain 실용 예제, GPT Builder

---

## 2. LangChain 기본 구조

### LangChain의 특징
LangChain은 LLM 애플리케이션 개발에 필요한 기능을 모듈화한 **"LLM Application Toolbox"**다.

- 모델, 프롬프트, 체인 연결
- 데이터베이스 및 메모리 연동
- Tool과 Agent 구성
- 높은 추상화를 통한 빠른 구현
- OpenAI, Anthropic, Google, Hugging Face, Ollama 등 지원

Provider별 라이브러리로 `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`, `langchain_huggingface`, `langchain_ollama` 등이 사용된다.

### LangChain 1.0 변화
2025년 10월 LangChain 1.0에서 라이브러리가 재편되었다.

- `langchain`: 주로 LLM Agent 개발 모듈
- `langchain-classic`: 기존 체인과 다양한 도구
- 구버전 코드가 작동하지 않을 경우 `langchain_classic`으로 변경하면 대부분 해결 가능

### Runnable 실행 모델
LangChain 컴포넌트는 기본적으로 `Runnable` 구조를 따른다.

- `invoke()`: 동기 실행
- `ainvoke()`: 비동기 실행
- `batch()`: 병렬 실행
- `abatch()`: 비동기 병렬 실행
- `stream()`: 순차 스트리밍 출력

특수 Runnable에는 다음이 있다.

- `RunnablePassthrough`: 이전 출력을 그대로 전달
- `RunnableParallel`: 여러 Runnable을 실행해 딕셔너리 형태로 반환
- `.assign()`: 기존 값에 실행 결과를 결합

이는 순차적 Chain을 보다 유연하게 구성하며, 이후 LangGraph의 그래프 구조로 확장될 수 있다.

### Structured Output
LLM 출력을 JSON, Pydantic, DateTime 등 구조화된 형식으로 변환하면 별도 후처리 없이 DB나 다른 프롬프트에 연결할 수 있다. 대표적으로 `with_structured_output()`을 활용한다.

---

## 3. RAG의 개념과 기본 파이프라인

### RAG가 필요한 이유
LLM은 사전학습된 언어 패턴을 바탕으로 그럴듯한 답변을 생성하지만 다음 영역에서는 정확성이 제한된다.

- 최신 정보
- 기업 내부 데이터
- 전문 도메인 데이터
- 학습되지 않은 사실

이 경우 **Hallucination(환각)** 위험이 커진다. RAG는 질문과 관련된 정보를 검색한 후 프롬프트에 함께 제공하여 LLM의 답변 정확도를 높인다.

### RAG의 5단계
1. **Indexing**: 문서를 검색 가능한 형태로 사전 구성
2. **Processing**: 사용자 질의를 검색에 적합하게 가공
3. **Searching**: 관련 문서 또는 청크 검색
4. **Augmenting**: 검색 결과와 질문을 프롬프트로 구성
5. **Generating**: LLM이 최종 답변 생성

---

## 4. Indexing과 문서 처리

### Chunking
LLM의 Context가 한정되어 있기 때문에 문서를 페이지, 문단, 토큰 단위 등으로 분할해 저장한다.

- 작은 청크: 검색 정밀도는 높지만 주변 맥락이 부족할 수 있음
- 큰 청크: 맥락은 풍부하지만 불필요한 정보가 포함되고 임베딩 정확도가 낮아질 수 있음
- Chunk overlap: 청크 경계에서 의미가 끊기는 문제를 완화

자료에 제시된 OpenAI 기본 RAG 예시는 **Chunk 800, Overlap 400**이다. 다만 최적 크기는 도메인과 데이터 특성에 따라 달라진다.

### 2026년 장문맥 관점
LLM의 Long Context 처리 능력이 향상되면서 청킹을 항상 수행해야 하는 것은 아니다.

- 문서가 **100k 미만**이라면 Full Context 사용이 더 안정적일 수 있음
- 큰 문서는 작은 청크보다 큰 청크를 사용해 맥락 단절을 줄이는 방향을 고려
- 단, 큰 청크의 검색 정확도를 높이기 위한 별도 기법이 필요

### 문서 로더와 파서
- LangChain Document Loader: PDF, DOCX, XML, Arxiv, PubMed, Tavily 등 지원
- **Docling**: PDF를 JSON/Markdown으로 변환하고 이미지·테이블을 추출하는 데 강점
- Docling은 GPU 실행 또는 `Docling-Serve` 기반 API 서빙을 권장

---

## 5. Query Processing과 검색

### 사용자 쿼리 처리
검색 전에 질문의 의도를 이해하고 적절히 재구성해야 한다.

- 어떤 데이터베이스를 검색할지 분류
- 짧은 질문은 이전 대화나 추가 정보를 반영해 Contextualize
- 긴 질문은 불필요한 내용을 제거하거나 요약
- 모호한 질문은 필요한 맥락을 추가해 재작성

### Dense Retrieval과 Sparse Retrieval
- **Dense/Semantic Search**: Transformer 기반 임베딩으로 의미적 유사성을 검색
- **Sparse/Lexical Search**: BM25, TF-IDF, SPLADE 등을 이용해 키워드 일치도를 검색

의미적으로 유사한 문서를 찾을 때는 Dense Retrieval이, 고유명사나 정확한 용어 일치가 중요할 때는 Sparse Retrieval이 유리하다.

### 벡터 데이터베이스
문서 청크의 임베딩을 저장한 후 질문 임베딩과 비교하여 Top-K 청크를 반환한다.

대표 제품:
- 전용 DB: Pinecone, Milvus, Qdrant, Chroma, Weaviate
- 기존 DB의 벡터 검색: Elasticsearch, OpenSearch

이미 기존 DB를 운영하고 있고 벡터 수가 많지 않다면 기존 DB의 벡터 검색 기능 활용을 권장한다.

주요 거리·검색 기준:
- Euclidean Distance
- Cosine Distance
- **MMR(Maximum Marginal Relevance)**: 유사성과 결과 다양성의 균형 고려

---

## 6. Advanced RAG 성능 개선

RAG 성능은 Chunk Size, Top-K, 검색 방식, 검색 결과의 순서와 품질 등 전체 파이프라인의 영향을 받는다.

### 6.1 Small2Big Chunking과 Sliding Window
작은 청크로 정밀하게 검색한 후 실제 프롬프트에는 주변의 큰 맥락을 함께 제공한다.

- Big Chunk와 Small Chunk의 2단 구성
- LangChain의 `ParentDocumentRetriever` 활용 가능
- Sliding Window로 검색 청크의 앞뒤 문맥 포함

이는 검색 정확도와 긴 Context 활용을 동시에 달성하는 **Retrieval-Generation Decoupling** 방식이다.

### 6.2 Contextual Retrieval
각 청크 앞부분에 전체 문서 관점의 설명을 추가해 다음 문제를 줄인다.

- 키워드 부재
- 상대적 표현
- 청크 시작점에서의 맥락 단절

Context Caching을 활용하면 비용을 절약할 수 있으며, Hybrid Search와 결합했을 때 성능 향상을 기대할 수 있다.

### 6.3 Query 확장 및 재구성
- **Multi-Query Retrieval**: 질문을 여러 버전으로 변형해 각각 검색
- **Query Reconstruction**: 추가 맥락을 반영해 검색 가능한 질문으로 재작성
- **HyDE(Hypothetical Document Embedding)**: LLM이 가상의 답변이나 관련 문서를 먼저 생성하고, 이를 임베딩하여 검색

### 6.4 Hybrid Search와 Metadata Filtering
- **Hybrid Search**: Lexical과 Semantic 검색 결과를 결합하며, 주로 Reciprocal Rank 기반으로 통합
- **Metadata Filtering**: 날짜, 작성자, 카테고리, 페이지 등의 메타데이터로 검색 범위를 제한

### 6.5 Context Optimization과 Reranking
Long Context에 많은 청크를 넣을 수 있어도 불필요한 정보는 성능을 저하시킨다.

- **Lost-in-the-middle Error**: 정답이 Context에 포함되어 있어도 중간에 위치하거나 주변 노이즈가 많으면 답변 성능이 저하되는 현상
- 중요 정보의 위치 조정
- 불필요한 청크 제거
- 검색 결과의 재정렬·필터링

Reranking에는 과거 BERT/T5 기반 Cross-Encoder가 사용되었으며, 현재는 LLM 자체로 관련성을 판정하고 재정렬하는 방법도 활용할 수 있다.

---

## 7. RAG 평가

Retrieval과 Generation을 분리해 평가해야 한다. 특히 Retrieval 평가는 중요하지만 정확한 측정을 위해 **Golden Data**가 필요하다.

### RAGAS 평가 프레임워크
RAGAS는 평가용 LLM과 임베딩 모델을 이용하여 RAG 품질을 측정한다.

주요 지표:
- **Faithfulness**: 답변이 제공된 Context에만 근거하는가
- **Factual Correctness**: 답변과 Reference의 사실·주장이 얼마나 일치하는가
- **Context Recall**: 검색 Context가 Reference의 내용을 충분히 포함하는가
- **BLEU Score**: 답변과 정답의 문자·표현 일치도
- **Semantic Similarity**: 답변과 정답 임베딩의 의미적 유사도

평가 결과를 바탕으로 청킹, 검색, 프롬프트, Reranking 등을 반복 개선해야 한다.

---

## 8. 2026년 RAG의 발전 방향

### 복잡한 문제 대응
단일 검색만으로 해결하기 어려운 문제에는 다음 기법이 필요하다.

- Multi-Query, HyDE 조합
- 여러 차례 검색하는 Deep Research 구조
- 특수 도메인 지식의 추가 학습
- GraphRAG
- 이미지·테이블을 처리하는 Multimodal RAG

### Agentic Retrieval
고정된 `Retrieval → Generation` 워크플로우는 검색 결과가 부족할 때 능동적으로 대응하기 어렵다. 향후에는 LLM이 다음을 판단하는 Agent 구조로 발전한다.

- 검색어 결정
- 검색 결과의 충분성 판단
- 추가 검색 여부 결정
- 다중 출처 검색
- 최종 답변 생성

즉, 고정 워크플로우는 약화되지만 **Retrieval과 효율적인 Context 구성의 중요성은 유지**된다.

### Retrieval의 4+1 유형
1. 벡터 DB 기반 Semantic 검색
2. 키워드 기반 검색
3. 카테고리·폴더 기반 결정적 검색
4. 그래프 기반 검색
5. 관계형 DB에 대한 SQL 검색

### LightRAG
LightRAG는 GraphRAG 계열의 대표적 오픈소스 구현으로 소개된다.

- 벡터 검색과 그래프 검색을 결합한 Dual Retrieval
- LLM 기반 Knowledge Graph 생성·검색
- 질문에 따라 5가지 쿼리 유형 지원
- Neo4j 없이 시작 가능
- EMNLP 2025 관련 기술

---

## 9. LangChain 실용 기능

### SQL 연동
`langchain_community`의 `SQLDatabase`와 SQL Query Chain을 이용하여 자연어로 SQL을 생성할 수 있다.

- SQL 자동 생성
- SQL 실행 체인 구성
- 실행 시 데이터 변조, 권한, 보안 문제를 반드시 고려

### Tool과 Agent
- **Tool**: 검색, API, 함수, 코드 실행 등 LLM이 호출할 수 있는 외부 기능
- **Agent**: Tool을 선택하고 실행 결과를 관찰하며 문제를 해결하는 모듈

대표 Tool:
- SerpAPI
- Tavily
- `PythonREPLTool`
- Decorator로 만든 Custom Tool

Tool Calling 과정:
1. 사용자가 요청
2. LLM이 Tool 호출 명령 생성
3. 프로그램이 Tool 실행
4. 결과를 Tool Message로 LLM에 전달
5. LLM이 최종 답변 생성

### ReAct Agent
ReAct는 **Thought–Action–Observation**을 반복하며 다음 행동을 계획하는 Agent 방식이다. LangChain 1.0의 `create_agent()`로 구현할 수 있다.

### MCP(Model Context Protocol)
MCP는 Tool Calling을 HTTP/STDIO 기반 Server–Client 구조로 일반화한 표준이다.

- 크로스플랫폼 지원
- 인증과 미들웨어 적용
- 외부 도구와 LLM의 연동 표준화

---

## 10. Claude Code Skills와 Context 관리

Tool과 MCP 설명이 지나치게 많아지면 **Context Bloat**가 발생한다. Anthropic이 2025년 10월 발표한 Skills는 필요한 기능만 점진적으로 불러오는 구조다.

### Skill 구성
- 이름과 짧은 설명
- `SKILL.md`
- 작업 지침과 프롬프트
- 실행 스크립트
- 예시 문서와 템플릿

`SKILL.md`는 YAML Frontmatter에 이름과 설명을 제공하고, Agent가 필요할 때 상세 내용을 읽어 Context에 추가하는 **Progressive Disclosure** 방식을 사용한다.

---

## 11. GPT Builder와 GPTs Action

OpenAI GPT Builder를 이용하면 별도 구현 부담을 줄이면서 커스텀 GPT를 만들 수 있다.

기본 지원 기능:
- Code Interpreter
- Retrieval
- Image Generation
- 외부 API 연동을 위한 Action

GPTs Action은 Action Schema를 정의하고 별도 서버를 구성하여 외부 API를 호출하는 방식이다.

---

## 12. 주요 실습

교육에서는 다음 실습을 수행한다.

- 여러 LLM Provider와 LangChain 연결
- `batch`, `stream`, Prompt Template 활용
- 데이터 분류·전처리 및 구조화된 데이터 생성
- ChromaDB와 웹페이지·뉴스 기반 RAG
- PDF 기반 연속 질의응답 및 `stuff`, `map-reduce`, `refine` 요약
- Advanced RAG 파이프라인 구현
- RAGAS 기반 RAG 평가
- SQL 자동 생성 및 실행
- Tavily·Custom Tool·Agent 연동
- Python Code Generator를 이용한 데이터 시각화
- GPT Builder로 커스텀 GPT 제작
- GPTs Action을 통한 외부 API 연결

## 핵심 결론

이 과정은 단순한 벡터 검색 기반 RAG 구현을 넘어, **장문맥 모델 시대의 Context 최적화, Hybrid Search, HyDE, Reranking, GraphRAG, Agentic Retrieval 및 체계적인 평가**까지 다룬다. 핵심 메시지는 RAG 성능이 LLM 자체보다도 **문서 구성, 질의 처리, 검색 정확도, Context 배치, 평가·개선의 전 과정**에 의해 결정된다는 점이다.
```

> ⚠️ **주목할 점**: 이 문서(`교재_0803.pdf`)는 **GPT Builder 챕터(원문 p.76~82)를 포함한 원본 PDF 그대로**이므로, Stuff 요약 결과에도 "11. GPT Builder와 GPTs Action" 섹션이 등장합니다. 이 README.md는 해당 챕터를 "더 이상 사용하지 않는 기술"로 판단하여 제외했지만(README 서두 "업데이트 내역" 참고), **원본 PDF 자체는 수정되지 않았으므로** 이 실습의 LLM 요약에는 GPT Builder 내용이 여전히 포함됩니다 — README(가공된 교재)와 원본 PDF(가공되지 않은 원문) 사이의 실제 차이를 보여주는 사례입니다.

---

### Part 4.2 Map-Reduce 방식

> Map Reduce 방식은 문서를 여러 청크로 나누어 처리합니다.
> 1. Map 단계: 각 청크를 개별적으로 요약
> 2. Reduce 단계: 개별 요약들을 합쳐서 최종 요약 생성
>
> **장점**: 매우 긴 문서도 처리 가능, 병렬 처리로 속도 향상 가능
> **단점**: 청크 간의 맥락이 손실될 수 있음, API 호출 횟수가 증가
>
> ```
> [청크1] → [요약1] ─┐
> [청크2] → [요약2] ─┼→ [최종 요약]
> [청크3] → [요약3] ─┘
> ```

```python
# Map 단계 프롬프트 (개별 청크 요약)
map_prompt = ChatPromptTemplate([
    ("system", """당신은 문서 요약 전문가입니다.
주어진 문서의 일부분을 읽고 핵심 내용을 요약해주세요.
이 요약은 나중에 다른 부분의 요약과 합쳐져 전체 요약이 됩니다."""),
    ("human", """다음 문서 일부를 요약해주세요:

---
{text}
---

핵심 내용을 간결하게 요약해주세요.""")
])

# Reduce 단계 프롬프트 (요약들을 합쳐서 최종 요약)
reduce_prompt = ChatPromptTemplate([
    ("system", """당신은 문서 요약 전문가입니다.
여러 부분의 요약들을 받아서 하나의 일관된 최종 요약을 작성해주세요.
중복되는 내용은 통합하고, 전체적인 흐름이 자연스럽게 연결되도록 해주세요."""),
    ("human", """다음은 문서의 여러 부분에 대한 요약들입니다:

---
{summaries}
---

위 요약들을 통합하여 전체 문서의 최종 요약을 작성해주세요.""")
])
```

```python
# Map 체인과 Reduce 체인
map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()
```

```python
# Map Reduce 실행
print("Map Reduce 방식으로 요약 중...")
start_time = time.time()

# Map 단계: 각 청크 요약
print(f"\n[Map 단계] {len(chunks)}개 청크 요약 중...")
chunk_summaries = map_chain.batch([{"text": c.page_content} for c in chunks])

print('개별 요약 처리 완료!')

# Reduce 단계: 요약들 통합
print("\n[Reduce 단계] 요약 통합 중...")
combined_summaries = "\n\n".join(chunk_summaries)
map_reduce_summary = reduce_chain.invoke({"summaries": combined_summaries})

elapsed_time = time.time() - start_time
print(f"\n완료! (소요 시간: {elapsed_time:.2f}초)\n")
print("="*60)
print("[Map Reduce 요약 결과]")
print("="*60)
print(map_reduce_summary)
```

**실행 결과** (소요 시간 43.56초, `map_chain.batch()`로 11개 청크를 병렬 요약한 후 통합):
```
Map Reduce 방식으로 요약 중...

[Map 단계] 11개 청크 요약 중...
개별 요약 처리 완료!

[Reduce 단계] 요약 통합 중...

완료! (소요 시간: 43.56초)

============================================================
[Map Reduce 요약 결과]
============================================================
## 최종 요약

이 문서는 2026년 삼성SDS **「RAG 개발 심화」 과정**의 교육 자료로, 변형호 강사가 진행한다. 과정의 목표는 LLM과 RAG의 핵심 원리를 이해하고, LangChain을 활용한 애플리케이션 개발부터 검색 성능 개선·평가, Tool·Agent, GPT Builder와 Actions까지 실습하는 것이다. 1일 차에는 LangChain 컴포넌트, 자동화·챗봇, 벡터스토어와 RAG를 다루고, 2일 차에는 RAG 고도화 및 평가, 실용 예제와 GPT Builder를 학습한다.

### 1. LangChain 기반 LLM 애플리케이션 개발

LangChain은 모델, 프롬프트, 체인, 데이터베이스, 메모리, 도구, 에이전트를 모듈식으로 연결하는 고수준 프레임워크다. OpenAI, Google, Hugging Face, Ollama 등 다양한 모델 제공자와 연동할 수 있으며, `init_chat_model()`과 System·Human·AI 메시지, 프롬프트 템플릿을 이용해 채팅 모델을 통합적으로 구성한다.

핵심 실행 단위는 `Runnable`이며 `invoke/ainvoke`, `batch/abatch`, `stream`을 통해 동기·비동기 실행, 병렬 처리와 스트리밍 출력을 지원한다. `RunnablePassthrough`, `RunnableParallel`, `.assign()`을 사용하면 데이터를 전달하거나 여러 작업을 병렬로 실행·결합할 수 있고, 이러한 체인 구조는 LangGraph의 그래프형 워크플로로 확장될 수 있다. `with_structured_output()`을 활용하면 LLM의 출력을 JSON, Pydantic, 날짜·시간 등 구조화된 형태로 받아 후속 처리에 바로 사용할 수 있다.

LangChain 1.0부터는 에이전트 중심의 `langchain`과 기존 기능을 제공하는 `langchain-classic`으로 분리되었다. 기존 코드가 동작하지 않을 때는 import를 `langchain_classic`으로 변경하거나 공식 마이그레이션 가이드를 참고해야 한다. 실습에서는 프롬프트 구성, 데이터 분류, 검색 체인, 성능 평가, 구조화 데이터 전처리와 소수 예제 기반 유사 데이터 생성을 수행한다.

### 2. RAG의 원리와 기본 파이프라인

LLM은 유연한 생성 능력을 갖지만 최신 정보, 내부 데이터, 전문 도메인 지식이 부족하면 환각이 발생할 수 있다. **RAG(Retrieval-Augmented Generation)**는 질문과 관련된 정보를 먼저 검색하고 이를 프롬프트에 명확히 구분해 제공함으로써, LLM의 In-Context Learning을 활용해 답변의 정확성과 근거성을 높이는 방식이다.

RAG는 일반적으로 다음 다섯 단계로 구성된다.

1. **Indexing**: 문서를 수집·분할하고 임베딩해 데이터베이스를 구성
2. **Processing**: 사용자 의도 분석과 질의 전처리·재구성
3. **Searching**: 질문과 관련된 문서 또는 청크 검색
4. **Augmenting**: 검색 결과를 포함한 프롬프트 구성
5. **Generating**: 검색 근거를 바탕으로 답변 생성

PDF, DOCX, XML과 Arxiv·Tavily·PubMed 등 다양한 파일 및 API는 LangChain Document Loader로 불러올 수 있다. Docling은 PDF를 JSON이나 Markdown으로 변환하고 이미지·표를 추출하는 데 강점이 있으며, GPU나 API 서버 형태로 운영할 수 있다.

### 3. 청킹·임베딩·검색 설계

문서는 LLM의 컨텍스트 제한과 검색 효율을 고려해 페이지, 문단, 문장 또는 토큰 단위로 나눈다. 작은 청크는 검색 정밀도가 높을 수 있지만 주변 맥락을 잃기 쉽고, 큰 청크는 맥락을 보존하지만 불필요한 정보가 섞여 검색과 임베딩 정확도가 저하될 수 있다. 청크 오버랩은 경계에서 발생하는 맥락 손실을 완화한다.

최근에는 장문맥 LLM의 성능 향상으로 고정된 작은 청크 전략에서 벗어나고 있다. 약 100k 이하의 작은 문서는 전체 문맥을 입력하는 방식이 더 안정적일 수 있으며, 대용량 문서는 맥락 단절을 줄이는 큰 청크와 이를 정확히 찾는 검색 전략이 필요하다. 장문맥 처리가 어려운 소형 LLM은 문제를 여러 단계로 나누어 RAG를 반복한 뒤 결과를 종합할 수 있다.

검색 방식은 크게 다음과 같이 나뉜다.

- **Dense Retrieval**: 임베딩을 이용한 의미 기반 검색
- **Sparse/Lexical Retrieval**: BM25, TF-IDF, SPLADE 등을 이용한 키워드 검색
- **Hybrid Search**: 의미 검색과 키워드 검색 결과의 결합
- **Metadata Filtering**: 카테고리, 날짜, 작성자, 페이지 등의 조건으로 범위 제한
- **Graph·SQL 검색**: 개체 관계나 관계형 데이터베이스를 활용한 검색

벡터 검색은 질문과 문서 청크의 임베딩 유사도를 계산해 상위 K개 결과를 반환한다. 거리 지표로는 유클리드 거리와 코사인 거리를 사용할 수 있으며, MMR은 관련성과 결과 다양성을 함께 고려한다. 대표적인 벡터 DB에는 Pinecone, Milvus, Qdrant, Chroma, Weaviate가 있다. 데이터가 작거나 Elasticsearch·OpenSearch를 이미 운영하고 있다면 기존 시스템의 벡터 검색 기능을 활용하는 것이 효율적이다.

### 4. RAG 성능 고도화

RAG 성능은 청크 크기, 오버랩, Top K, 검색 방식과 가중치, 결과 순서 등 파이프라인 전체 설계에 좌우된다. 주요 개선 방법은 다음과 같다.

- **Small2Big·Sliding Window**: 작은 청크로 정확하게 검색한 뒤 연결된 큰 청크나 주변 문맥을 생성 단계에 제공
- **Contextual Retrieval**: 전체 문서를 반영한 설명을 각 청크 앞에 추가해 키워드 누락, 상대 표현, 문장 단절 문제를 완화하며 캐싱으로 비용 절감
- **Multi-Query Retrieval**: 하나의 질의를 여러 형태로 변환해 검색 범위 확장
- **Query Reconstruction**: 짧은 질문에는 대화 이력과 추가 맥락을 보완하고, 긴 질문에서는 불필요한 내용을 제거·요약
- **HyDE**: 모호한 질문에 대해 가상의 답변이나 관련 문서를 생성한 뒤 이를 임베딩해 검색
- **Reranking**: Cross-Encoder나 LLM으로 질문과 검색 문서의 관련성을 다시 평가해 재정렬·필터링
- **Context Optimization**: 중요 정보를 앞이나 뒤에 배치하고 불필요한 청크를 제거

긴 컨텍스트에 많은 자료를 넣더라도 주변 정보가 노이즈로 작용하거나 중간에 있는 핵심 정보를 놓치는 **Lost-in-the-middle** 문제가 발생할 수 있다. 따라서 단순히 더 많은 문서를 제공하기보다 검색 결과의 품질, 순서, 관련성 및 컨텍스트 구성이 중요하다.

실습에서는 ChromaDB와 웹·뉴스·PDF 데이터를 이용한 RAG, 연속 질의응답, 그리고 `stuff`, `map-reduce`, `refine` 방식의 문서 요약을 구현한다.

### 5. RAG 평가와 발전 방향

RAG는 검색과 생성을 분리해 평가해야 한다. 검색 정확도를 신뢰성 있게 측정하려면 정답 문서가 정의된 Golden Data가 필요하며, RAGAS 등의 도구로 다음 지표를 평가할 수 있다.

- Faithfulness
- Factual Correctness
- Context Recall
- BLEU
- Semantic Similarity

실습에서는 RAG 파이프라인에 RAGAS 평가 모듈을 연결하고, 평가 프롬프트와 검색·생성 방법을 개선하면서 성능을 비교한다.

현대 RAG는 단순한 **검색→생성** 구조에서 벗어나 검색 결과가 부족하면 추가 검색과 다중 출처 조사를 수행하는 **Retrieval Agent**로 발전하고 있다. Multi-Query, HyDE, Deep Research뿐 아니라 특수 도메인 학습, GraphRAG, Multimodal RAG가 임베딩 검색의 한계를 보완한다. LightRAG는 LLM 기반 지식 그래프 생성, 벡터·그래프 듀얼 검색, 질문별 쿼리 유형 전환을 지원하며 별도 Neo4j 없이도 시작할 수 있다.

### 6. SQL, Tool, MCP와 Agent

LangChain의 `SQLDatabase`와 SQL Query Chain을 이용하면 자연어 질문에서 SQL을 자동 생성할 수 있다. 다만 쿼리 실행 체인은 별도로 구성해야 하며, 데이터 접근 권한, 위험한 명령 차단, 입력 검증 등 보안과 안전성을 고려해야 한다.

**Tool**은 검색 API, Python 실행, 파일 저장, 서버 상태 조회 등 LLM이 외부 기능을 사용하게 하는 모듈이다. LLM이 도구 호출 명령과 인자를 생성하면 애플리케이션이 이를 실행하고, 실행 결과를 다시 LLM에 전달해 최종 답변을 만든다. LangChain은 SerpAPI, Tavily, `PythonREPLTool` 등을 지원하며, 사용자 정의 함수도 설명과 인자 정보를 지정하고 데코레이터를 적용해 Tool로 변환할 수 있다.

**MCP(Model Context Protocol)**는 Tool 연동을 HTTP 또는 STDIO 기반 서버·클라이언트 구조로 표준화해 플랫폼 간 호환성과 인증·미들웨어 기능을 제공한다.

**Agent**는 도구를 한 번 호출하는 데 그치지 않고, 계획 수립→도구 실행→결과 관찰을 반복하며 복잡한 문제를 해결한다. 대표적인 ReAct Agent는 '생각–행동–관찰' 과정을 통해 다음 행동을 결정한다. 실습에서는 `create_agent()`를 이용해 Tavily Search와 사용자 정의 Tool을 연결한다.

### 7. Skills, GPT Builder와 Actions 실습

Claude Code Skills는 많은 도구 설명을 한꺼번에 제공해 발생하는 **Context Bloat**를 줄이기 위해 점진적 공개 방식을 사용한다. 처음에는 스킬 이름과 짧은 설명만 제공하고, 실제로 선택되었을 때 상세 지침, 예시, 실행 스크립트, 템플릿과 링크를 컨텍스트에 추가한다.

Skill은 단일 파일 형식이 아니라 `SKILL.md`, 작업용 프롬프트, 스크립트, 예시 문서 등으로 구성되는 폴더 구조다. `SKILL.md`의 YAML Frontmatter에는 이름과 설명을 기록하고, 본문에는 사용 지침과 예시 등을 담는다. RunPod Manager 예시에서는 GPU Pod의 일괄 생성·조회·중지·삭제, 루트 비밀번호와 SSH·Jupyter 설정 자동화, 접속 정보의 CSV 출력을 구현한다.

최종 실습에서는 LangChain과 Python Code Generator를 결합해 데이터 시각화 코드를 생성·실행·평가·개선하고, 스킬 실행 과정을 익힌다. 또한 GPT Builder의 Code Interpreter, Retrieval, 이미지 생성 기능으로 커스텀 GPT를 제작하며, GPTs Action Schema와 별도 API 서버를 이용해 외부 서비스를 연동한다.

종합하면 이 과정은 LangChain의 기본 체인 구성에서 출발해 RAG의 데이터 처리·검색·생성·평가를 체계적으로 다루고, 장문맥·하이브리드 검색·GraphRAG·Retrieval Agent와 같은 최신 방향까지 확장한다. 궁극적으로는 Tool, MCP, Agent, Skills, GPT Actions를 결합해 검색과 외부 기능을 자율적으로 활용하는 실용적인 LLM 애플리케이션을 구축하는 데 초점을 둔다.
```

> 💡 **Stuff vs Map-Reduce 비교**: 두 결과 모두 원문의 내용을 충실히 담고 있지만, 세부 절 구성과 표현이 다릅니다. Stuff 요약은 원문 전체를 한 번에 본 LLM이 **자신만의 12개 섹션 구조**로 재구성했고, Map-Reduce 요약은 **11개 청크별 요약을 다시 통합**한 결과라 원문의 슬라이드 순서(1~7번 섹션)에 더 가깝게 따라갑니다. 소요 시간은 Stuff(40.07초, LLM 호출 1회) vs Map-Reduce(43.56초, LLM 호출 12회=Map 11회+Reduce 1회)로 이번 실습 규모(11개 청크)에서는 큰 차이가 없었지만, 문서가 훨씬 커지면 Map 단계가 병렬화(`batch()`)되어 있어 Map-Reduce가 유리해집니다.

---

### Part 4.3 Refine 방식

> Refine 방식은 청크를 순차적으로 처리하며 요약을 점진적으로 개선합니다.
> 1. 첫 번째 청크로 초기 요약 생성
> 2. 다음 청크와 현재 요약을 함께 보고 요약 개선
> 3. 모든 청크를 처리할 때까지 반복
>
> **장점**: 문서의 맥락을 유지하며 요약, 순차적으로 정보가 누적됨
> **단점**: 순차 처리로 시간이 오래 걸림, 앞부분 내용이 뒷부분에 의해 희석될 수 있음
>
> ```
> [청크1] → [요약v1]
>               ↓
> [청크2] + [요약v1] → [요약v2]
>                         ↓
> [청크3] + [요약v2] → [최종 요약]
> ```

```python
# 초기 요약 프롬프트 (첫 번째 청크용)
initial_prompt = ChatPromptTemplate([
    ("system", """당신은 문서 요약 전문가입니다.
문서의 첫 부분을 읽고 초기 요약을 작성해주세요.
이 요약은 이후 문서의 다른 부분을 읽으면서 점진적으로 개선될 것입니다."""),
    ("human", """다음은 문서의 첫 부분입니다:

---
{text}
---

이 내용을 바탕으로 초기 요약을 작성해주세요.""")
])

# 개선 프롬프트 (후속 청크용)
refine_prompt = ChatPromptTemplate([
    ("system", """당신은 문서 요약 전문가입니다.
기존 요약과 새로운 문서 부분을 함께 보고, 요약을 개선해주세요.
새로운 정보가 있다면 추가하고, 기존 내용과 통합하여 일관된 요약을 만들어주세요."""),
    ("human", """현재까지의 요약:
{current_summary}

---

새로운 문서 부분:
{text}

---

위의 새로운 내용을 반영하여 요약을 개선해주세요.""")
])
```

```python
# Refine 체인들
initial_chain = initial_prompt | llm | StrOutputParser()
refine_chain = refine_prompt | llm | StrOutputParser()
```

```python
# Refine 실행
print("Refine 방식으로 요약 중...")
start_time = time.time()

# 첫 번째 청크로 초기 요약 생성
print(f"\n[초기 요약] 청크 1/{len(chunks)} 처리 중...")
current_summary = initial_chain.invoke({"text": chunks[0].page_content})
print(f"  청크 1/{len(chunks)} 완료")

# 나머지 청크들로 요약 개선
for i, chunk in enumerate(chunks[1:], start=2):
    print(f"  청크 {i}/{len(chunks)} 처리 중...")
    current_summary = refine_chain.invoke({
        "current_summary": current_summary,
        "text": chunk.page_content
    })
    print(f"  청크 {i}/{len(chunks)} 완료")

refine_summary = current_summary

elapsed_time = time.time() - start_time
print(f"\n완료! (소요 시간: {elapsed_time:.2f}초)\n")
print("="*60)
print("[Refine 요약 결과]")
print("="*60)
print(refine_summary)
```

**실행 결과** (노트북에 실제로 기록된 그대로):
```
Refine 방식으로 요약 중...

[초기 요약] 청크 1/11 처리 중...
  청크 1/11 완료
  청크 2/11 처리 중...
  청크 2/11 완료
  청크 3/11 처리 중...
  청크 3/11 완료
  청크 4/11 처리 중...
  청크 4/11 완료
  청크 5/11 처리 중...
  청크 5/11 완료
  청크 6/11 처리 중...
  청크 6/11 완료
  청크 7/11 처리 중...
  청크 7/11 완료
  청크 8/11 처리 중...
  청크 8/11 완료
  청크 9/11 처리 중...
```

> ⚠️ **실제로 발견된 이슈 — 출력이 중간에 끊김**: 노트북에 저장된 이 셀의 실행 결과는 **"청크 9/11 처리 중..."에서 끝나 있고, 청크 10·11 처리 로그와 최종 `[Refine 요약 결과]`가 기록되어 있지 않습니다.** 셀의 `execution_count`도 비어 있어(`exec=None`), 이는 다음 중 하나로 추정됩니다.
> - Colab 세션이 실행 도중 끊기거나(연결 유실), 런타임이 재시작되어 출력이 끝까지 저장되지 못한 경우
> - 순차 처리(총 11회 LLM 호출) 특성상 실행 시간이 길어, 마지막까지 실행되기 전에 노트북이 저장된 경우
>
> **이는 이 교재에서 임의로 만들어내지 않고 있는 그대로 밝히는 실제 한계**입니다 — Refine 방식의 "단점"(순차 처리로 시간이 오래 걸림)이 실습 도중 실제로 체감된 사례라고 볼 수 있습니다. Stuff(40초)·Map-Reduce(44초, 병렬)에 비해, Refine은 **11번의 LLM 호출을 순서대로(직렬로) 기다려야 하므로 셋 중 가장 느립니다** — 청크 9개를 처리하는 데만 이미 스톱워치가 계속 돌고 있었다는 것이 바로 이 방식의 근본적인 트레이드오프를 보여줍니다.

### Part 4.4 세 가지 방식 비교 (노트북 원문 정리표)

| 방식 | 특징 | 적합한 상황 |
|------|------|-------------|
| Stuff | 전체를 한 번에 처리 | 짧은 문서, 빠른 결과 필요시 |
| Map Reduce | 병렬 처리 후 통합 | 긴 문서, 병렬 처리 가능시 |
| Refine | 순차적 개선 | 맥락 유지가 중요한 경우 |

---

## Part 5. [실습 과제] 임의의 PDF 다운로드하여 요약하기

> arxiv 등의 페이지에서 PDF를 다운로드하여 업로드하고, Stuff/Map-Reduce/Refine 등의 방법을 이용해 전체 PDF를 요약하세요.

```python
# (이 셀은 노트북에 비어 있는 상태로 남아 있습니다 — 실습 과제이므로 학습자가 직접 작성)

```

```python
# (이어지는 두 번째 실습 셀도 비어 있는 상태입니다)

```

> ℹ️ 원본 노트북에는 이 실습 과제에 대한 코드가 채워지지 않은 채로 남아 있습니다(두 개의 빈 코드 셀). 이 교재는 실습 결과를 있는 그대로 반영한다는 원칙에 따라, 실제로 작성되지 않은 코드를 임의로 만들어 채우지 않았습니다. 이 실습을 직접 수행하려면 Part 4에서 구현한 `stuff_chain`/`map_chain`+`reduce_chain`/`initial_chain`+`refine_chain`을 그대로 재사용하여, `PyMuPDFLoader`로 새 PDF를 로드하고 동일한 파이프라인에 흘려보내면 됩니다.

---

## Part 6. 정리

### 6.1 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | PDF 로드(`PyMuPDFLoader`) | 82페이지 전체 로드, `langchain_community` Deprecation 경고 확인 |
| 2 | 토큰 기반 청킹 | 14,592자 → 11개 청크(`chunk_size=1000, overlap=200`, tiktoken 기준) |
| 3 | Chroma 벡터스토어 + Retriever | Top-5 검색에서 "LangChain의 장점은?" 질문의 정답 청크가 상위권에 명확히 잡히지 않는 한계 확인 |
| 4 | RAG 질의응답 체인 | 흩어진 단서를 종합해 실질적으로 정확한 답변 생성(40초 미만) |
| 5 | Stuff 요약 | 전체 문서 1회 호출, 40.07초, 12개 섹션으로 재구성된 요약 |
| 6 | Map-Reduce 요약 | 11개 청크 병렬 요약 + 통합, 43.56초 |
| 7 | Refine 요약 | 순차 11회 호출 중 **청크 9/11에서 기록 중단**(실제 발생한 미완료 사례) |
| 8 | [실습 과제] | 원본 노트북에 미작성 상태로 남음(학습자 몫) |

### 6.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 확인한 것 |
|---|---|
| RAG의 5-step Process(p.25) | PDF 로드(Indexing) → 청킹 → Chroma 저장 → Retriever(Searching) → Prompt(Augmenting) → LLM(Generating) 전 과정을 그대로 구현 |
| Indexing: 청킹에 대한 견해 변화(p.28) | 14,592자(100k 미만)인 문서에도 토큰 기반 청킹을 적용해보고, Stuff 방식(청킹 없이 전체 투입)과 비교 |
| LangChain Document Loader(p.29) | `PyMuPDFLoader`로 PDF를 페이지 단위 `Document` 리스트로 로드 |
| Searching: 벡터DB(p.32~36) | Chroma + `text-embedding-3-large`로 Top-K(k=5) 시맨틱 검색 구현 |
| Advanced RAG (Augmenting) — Long Context/Lost-in-the-middle(p.47~48) | "Top-K 기반 RAG의 한계"를 실제로 체감하고, 이를 보완하는 요약 전략(Stuff/Map-Reduce/Refine)으로 이어짐 |
| LangChain 1.0(p.11) | `langchain_community` Deprecation 경고를 실제로 확인 |

### 6.3 참고 자료

- LangChain Document Loaders: <https://python.langchain.com/docs/integrations/document_loaders/>
- Chroma: <https://www.trychroma.com/>
- tiktoken (OpenAI 토크나이저): <https://github.com/openai/tiktoken>
- Lost in the Middle 논문: <https://arxiv.org/pdf/2307.03172>
- 원본 실습 노트북: `[실습]_5_PDF_내용_기반_질의응답_어플리케이션.ipynb`

### 6.4 다음 단계

- **Lab 4** (`[실습]_4_...ipynb`, README p.37)에서는 `WebBaseLoader()`와 ChromaDB로 웹페이지·뉴스 기사 기반 RAG를 다룹니다 — 이번 실습의 PDF 대신 웹 문서를 다루는 자매 실습입니다.
- **Lab 6** (`[실습]_6_Advanced_RAG+RAG성능평가.ipynb`)에서는 이번 Part 4에서 확인한 "Top-K 검색의 한계"를 **Multi-Query / Ensemble(Hybrid) / Rerank / Contextual Retrieval / Agentic RAG**로 정면으로 해결합니다. 특히 Contextual Retrieval은 이번 실습의 "청크가 문맥을 놓치는 문제"에 대한 구체적 해법입니다.
- 실무 팁: 문서가 작을 때는 이번 실습처럼 Stuff 요약이 충분히 실용적입니다. 문서량이 늘어나 Context Length를 초과할 것 같으면 Map-Reduce(속도 우선, 병렬화 가능) 또는 Refine(맥락 유지 우선, 순차 처리) 중 요구사항에 맞는 방식을 선택하십시오 — 이번 실습의 Refine 미완료 사례처럼, **순차 처리 방식은 청크 수가 많아지면 실행 시간과 안정성(연결 유지) 리스크가 함께 커진다는 점**을 고려해야 합니다.
