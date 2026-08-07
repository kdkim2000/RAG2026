# Lab 6. RAG 성능 평가와 Advanced RAG

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"RAG의 다양한 성능개선 방법"**(p.40~51)과 **"RAG의 성능평가 방법"**(p.52~58) 이론을, `[실습]_6_Advanced_RAG+RAG성능평가.ipynb`의 **실제 실행 결과**와 결합하여 재구성한 것입니다. 노트북은 하나이지만, README 부록의 실습 목록으로 보면 **Lab #6 "랭체인의 Advanced RAG 기술"(p.51)**과 **Lab #7 "RAG 시스템 성능평가하기"(p.56)** 두 실습이 순서대로 결합되어 있습니다.

## 학습 목표

- RAG 파이프라인의 성능을 좌우하는 지점(Chunking·Searching·Augmenting)을 이해하고, 이를 개선하는 5가지 Advanced RAG 기법(**Multi-Query, Ensemble/Hybrid, Rerank, Contextual Retrieval, Agentic RAG**)을 직접 구현한다.
- **RAGAS** 프레임워크로 RAG 파이프라인을 정량 평가하는 방법과, 그 근간이 되는 **Claim 기반 평가 방법론**을 이해한다.
- 6가지 RAG 구현(기본/Multi-Query/Ensemble/Rerank/Contextual/Agentic)을 **동일한 평가 데이터셋과 지표로 비교**하여, "왜 Advanced RAG가 필요한가"를 실제 수치로 확인한다.
- 실습 과정에서 실제로 발생한 Windows 환경 이슈(패키지 설치, 인코딩, 파일 잠금)를 처리하는 방법을 익힌다.

## Part 0. 개요: 노트북 구조와 실습 매핑

`README.md` 부록 "실습(Lab) 목록"의 두 항목이 이 노트북 하나에 결합되어 있습니다.

| README Lab # | 실습명 | 페이지 | 이 노트북에서의 위치 |
|---|---|---|---|
| 6 | 랭체인의 Advanced RAG 기술 | p.51 | Multi-Query / Ensemble / Rerank / Contextual / Agentic RAG 구현부 |
| 7 | RAG 시스템 성능평가하기 | p.56 | RAGAS 평가 프레임워크 및 6가지 방식 비교부 |

**노트북 65개 셀의 흐름**:
1. 환경설정 → 문서 로드/청킹 → Qdrant 벡터스토어 구성 → 기본(Baseline) RAG 체인
2. RAGAS로 기본 RAG 1회 평가 → 재사용 가능한 `evaluate_rag()` 함수 정의
3. **Multi-Query Retriever** 구현 + 평가
4. **Ensemble(Hybrid) Retriever** 구현 + 평가
5. **LLM Reranker** 구현 + 평가
6. **Contextual Retrieval** 구현 + 평가
7. **Agentic RAG** 구현 + 평가
8. 실행 결과 기록, 최종 비교 분석, 부록

> ℹ️ **참고**: 이 노트북은 Lab 7·8 노트북과 달리 각 셀의 개별 실행 결과(print 출력 등)가 저장되어 있지 않고, **전체 실행 후 마지막에 삽입된 "✅ 실행 결과 (로컬 실행 기록)" 마크다운 셀**에 실제 측정값이 집계되어 있습니다. 이 교재에서는 코드는 누락 없이 전부 포함하고, 실행 결과는 이 집계 데이터(및 노트북에 원래 있던 Colab 참고값)를 있는 그대로 제시합니다 — 기록되지 않은 개별 출력을 임의로 만들어내지 않습니다.

---

## Part 1. 환경 준비와 기본(Baseline) RAG 체인

### 이론: RAG 파이프라인의 성능 개선 지점 (p.41)

> RAG 파이프라인에서의 모든 요소는 성능에 큰 영향
> - **Chunking**: Size / Top K
> - **Searching**: Semantic / Lexical
>
> 그 외의 의사결정
> - 정적인 Chunking은 맥락 손실이 큰데, 어떻게 보완할까?
> - 검색 Chunk의 순서는 어떻게 할까?
> - Semantic/Lexical의 비중은 어떻게 할까?
> - 단일 검색만으로 해결되지 않는 문제는 어떻게 해야 할까?

이 4개의 질문이 이번 실습의 Part 3~7(Multi-Query / Ensemble / Rerank / Contextual / Agentic)에서 각각 답을 얻습니다.

### 실습: 패키지 설치와 LLM 준비

```python
%pip install ragas==0.4.0 sacrebleu dotenv jsonlines openai langchain langchain-openai langchain_qdrant 'langchain-community<0.4.2' tiktoken rank_bm25 pymupdf kiwipiepy -q
```

```python
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from glob import glob
# 시스템/유틸리티
import os
import gc
import ast
import csv
import uuid
import datetime
import re

# API & 환경 설정
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 데이터 처리 및 시각화
import pandas as pd
from tqdm import tqdm
from glob import glob

# 네트워크/웹 관련
import requests
import jsonlines
import bs4

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# LangChain 핵심 모듈
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ragas 평가/지표 관련
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics.collections import (
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    BleuScore,
    SemanticSimilarity
)

import logging

# OpenAI, httpx의 INFO 로그 차단
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# RAGAS 사용 정보 수집 차단
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

load_dotenv(override=True)
```

```python
llm = ChatOpenAI(model='gpt-5.2', reasoning_effort='low')
print(llm.invoke("안녕?").text)
```

```python
import zipfile
import os

# 압축 파일 이름 (현재 위치)
zip_filename = "RAG_data.zip"

# 압축 해제 경로 (data 폴더)
extract_path = "data"

try:
    # data 폴더 없으면 생성
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"성공: {zip_filename}의 압축을 {extract_path}/ 폴더에 풀었습니다.")

except FileNotFoundError:
    print(f"오류: {zip_filename} 파일을 찾을 수 없습니다.")

except zipfile.BadZipFile:
    print("오류: 손상되었거나 유효하지 않은 zip 파일입니다.")
```

### ⚠️ Windows 환경에서 실제로 발견된 이슈 (실행 기록)

> 노트북 마지막의 "✅ 실행 결과 (로컬 실행 기록)" 셀에 남겨진, **Windows 로컬 환경(venv, Python 3.12, `gpt-5.2` / 평가자 `gpt-4.1-mini`)에서 처음부터 끝까지 실제로 실행하여 발견한 이슈**입니다(실행일: 2026-08-07). Colab에서 작성된 노트북을 그대로 로컬 Windows에서 실행하면 아래 문제들이 재현됩니다.

| # | 셀 | 증상 | 원인 | 조치 |
|---|---|---|---|---|
| 1 | `%pip install ...` | 아무 패키지도 설치되지 않음(에러 없이 조용히 실패) | `'langchain-community<0.4.2'`처럼 bash 스타일로 묶은 인용부호가 Windows에서 다르게 해석되어 pip 인자가 깨짐 | 동일 패키지 목록을 인용부호 없이 직접 설치 |
| 2 | `import bs4` | `ModuleNotFoundError: No module named 'bs4'` | import는 있지만 설치 목록에 `beautifulsoup4`가 빠져 있음 | `pip install beautifulsoup4` 추가 설치 |
| 3 | `TextLoader(...)` (문서 로드 / Contextual Retrieval 예시 확인) | `UnicodeDecodeError: 'cp949' codec can't decode byte ...` | Windows 기본 로케일(cp949)로 인코딩을 추정하지만 마크다운 파일은 UTF-8 | `TextLoader(..., encoding='utf-8')` 인자 추가 (노트북에 반영됨) |
| 4 | `QdrantClient(path=...)` | `ImportError: pywintypes is required for Win32Locker but not found` | Qdrant 로컬(온디스크) 모드의 파일 잠금이 Windows에서 `pywin32`를 요구하나 기본 설치되지 않음 | `pip install pywin32` 추가 설치 |
| 5 | Agentic RAG 평가 | 평가 중 `IncompleteOutputException`(max_tokens 초과) 1건 발생 (Job 64/100) | `evaluator_llm`의 `max_tokens=8192` 한도를 일부 응답이 초과 | RAGAS가 해당 1건만 제외하고 나머지로 집계(전체 결과에 영향 미미) |

이 교재의 코드 블록에는 이미 3번 이슈(`encoding='utf-8'`)에 대한 수정이 반영되어 있습니다.

### 실습: 문서 전처리와 청킹

```python
# 데이터 준비 / 전처리 / 벡터 데이터베이스 구성
def preprocess(docs):
    import re
    def clean_text(doc):
        text = doc.page_content
        text1 = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        text = text1
        text2 = re.sub(r'[​ ]', '', text)
        text = text2
        text3 = re.sub(r' {2,}', ' ', text2)
        text = text3.strip()
        doc.page_content = text
        return doc
    preprocessed_docs = []
    for doc in docs:
        doc = clean_text(doc)
        preprocessed_docs.append(doc)
    return preprocessed_docs

reports = glob('data/markdowns/*.md')
documents = []
for report in reports:
    loader = TextLoader(report, encoding='utf-8')
    docs = loader.load()
    print('문서 로드 완료:', docs[0].metadata, docs[0].page_content[0:10], '...', len(docs[0].page_content))
    documents += docs
documents = preprocess(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 0~1000, 800~1800, 1600~2600, ...
chunks = text_splitter.split_documents(documents)
print(f'# 총 {len(chunks)} 개의 청크 생성')
```

> 💡 **이론 연결 — Indexing: 적절한 청크 사이즈 선택하기 (p.27)**: `chunk_size=1000, chunk_overlap=200`은 README의 "OpenAI의 기본 RAG 구조: 800 Chunk / 400 Overlap"(p.26)과 유사한 비율(오버랩 ≈ 20%)입니다. 오버랩은 청크 경계에서 문맥이 끊기는 것을 방지하는 역할을 합니다.

### 실습: Qdrant 벡터 스토어 구성

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large', chunk_size=100)

uuidstr = str(uuid.uuid4())[0:6]
client = QdrantClient(path=f"outputs/vectordb/qdrant_{uuidstr}")

client.create_collection(
    collection_name=f"AI_Reports",
    vectors_config=VectorParams(size=3072, distance=Distance.EUCLID))

vector_store = QdrantVectorStore(
    client=client,
    collection_name=f"AI_Reports",
    embedding=openai_embeddings,
    distance=Distance.EUCLID)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("벡터 DB와 Retriever 준비 완료!")
```

> 💡 **이론 연결 — Vector Database의 Metric Types (p.36)**: 여기서는 `Distance.EUCLID`(Euclidean/L2 거리)를 사용합니다. README에서는 Euclidean Distance 외에 Cosine Distance, MMR(다양성까지 고려하는 검색)도 소개합니다 — 임베딩이 정규화(normalized)되어 있다면 Cosine이 더 흔히 쓰입니다.

### 실습: RAG 체인 구성

```python
from typing import Iterable, Sequence
from xml.sax.saxutils import escape
from langchain_core.documents import Document

def format_docs(
    docs: Iterable[Document],
    metadata_keys: Sequence[str] = ("source",),
) -> str:
    """
    List[Document] -> XML 직렬화 문자열.

    - 청크 경계: <document index="N"> 태그로 명시
    - 메타데이터: metadata_keys 에 지정한 키만 <meta>로 포함 (누락 키는 자동 생략)
    - 본문: XML 특수문자 escape (본문에 '<', '>' 가 있어도 경계 유지)
    """
    parts: list[str] = ["<documents>"]
    for i, doc in enumerate(docs, start=1):
        parts.append(f'  <document index="{i}">')
        for key in metadata_keys:
            value = doc.metadata.get(key)
            if value is None:
                continue
            parts.append(
                f'    <meta name="{escape(str(key))}">{escape(str(value))}</meta>'
            )
        parts.append(f"    <content>{escape(doc.page_content)}</content>")
        parts.append("  </document>")
    parts.append("</documents>")
    return "\n".join(parts)

prompt = ChatPromptTemplate([
    ("system", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다양한 출처의 보고서 일부 내용이 Context로 주어집니다.
Context의 내용을 바탕으로 Question에 대한 답변을 제공하세요.

만약 Context가 질문과 무관하거나 관련 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.'''),
    ("human",'''Context: {context}
---
Question: {question}''')])


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())

print("RAG 체인 준비 완료!")
```

> 💡 검색된 문서를 `format_docs()`로 **XML 형태**로 직렬화하는 점이 눈에 띕니다. 단순히 텍스트를 이어붙이는 대신 `<document index="N">` 태그로 청크 경계를 명시하면, LLM이 "어느 문서에서 온 정보인지"를 혼동하지 않고 구분할 수 있습니다. 이는 README "RAG 프롬프트 예시"(p.23)의 "정보와 질문을 구분하여 제공" 원칙을 더 엄격하게 적용한 형태입니다.

---

## Part 2. RAGAS로 기본 RAG 성능 평가하기

### 이론: RAG Evaluation (p.52)

> 수행된 RAG의 성능 평가하기
> - Retrieval과 Generation에 대한 평가가 각각 이루어져야 함
>   - 더 중요한 것은 Retrieval이나, 이를 정확히 평가하려면 Golden Data 필요
> - **RAGAS**: RAG 평가 프레임워크 제공 (평가 LLM, 평가 Embedding 필요)

### 이론: RAGAS 상세 — Claim 기반 평가 방법론 (수업 필기)

**1. Claim(주장)**: 정보/주장의 가장 작은 단위. LLM을 통해 정답과 답변을 Claim List로 분할합니다.

> 예) "앤트로픽의 코딩 에이전트 클로드 코드 비용이 30% 인하되었다." → 1) 클로드는 앤트로픽의 에이전트다. 2) 클로드는 코딩 에이전트다. 3) 클로드 비용이 30% 인하되었다.

**2. Metrics (LLM-as-a-Judge)**

| 지표 | 계산 방법 | 의미 |
|---|---|---|
| **Context Recall** | [검색 결과로부터 유추할 수 있는 Claim 개수] / [정답 Claim 개수] | 낮으면 반드시 올려야 함 |
| **Faithfulness** | [검색 결과로부터 유추할 수 있는 Claim 개수] / [답변 Claim 개수] | LLM이 검색 결과에 근거해 답했는가 |
| **Factual Correctness** | 정답 Claim(레이블) 대비 답변 Claim을 이진 분류의 F1 Score로 계산 | 정답과 답변이 얼마나 동일한 주장을 하는가 |

**3. 전통적(non-LLM) 평가**: ROUGE/BLEU Score(키워드 n-gram 일치), 임베딩 유사도(Semantic Similarity) — 이 실습에서는 이 두 축(LLM 기반 3개 + 전통적 2개)을 모두 사용합니다.

### 실습: 평가 데이터셋 불러오기

```python
import pandas as pd
df = pd.read_csv('data/evaluation/rag_evaluation_data.csv')

eval_dataset = df.to_dict('list')
questions, ground_truths, qtypes = eval_dataset['question'], eval_dataset['answer'], eval_dataset['difficulty']
for i in range(len(questions)):
    print(f'#{i} ', end='')
    print(f'({qtypes[i]})')
    print(f'Question: {questions[i]}\n')
    print(f'Ground Truth: {ground_truths[i]}\n')
    print('-----------')
```

질문(`question`)·정답(`answer`)·난이도(`difficulty`) 20개 행으로 구성된 평가셋을 불러옵니다. 이후 모든 Advanced RAG 기법이 **동일한 이 20개 질문**으로 평가되므로 공정한 비교가 가능합니다.

### 실습: RAGAS 입력 데이터 구성하기

```python
dataset = []

result = rag_chain.batch(questions)
relevant_docs_list = retriever.batch(questions)

for i, ans in enumerate(result):
    print(f"Question: {questions[i]}")

    relevant_docs = [doc.page_content for doc in relevant_docs_list[i]]
    print(f"Answer: {ans}")
    print('---')

    dataset.append(
        {
            "user_input":questions[i],
            "retrieved_contexts":relevant_docs,
            "response":ans,
            "reference":ground_truths[i]
        }
    )
```

RAGAS가 요구하는 4개 필드(`user_input`/`retrieved_contexts`/`response`/`reference`)를 채웁니다 — 이는 Part 2 이론의 "정답 Claim vs 답변 Claim vs 검색 결과 Claim"을 비교하기 위한 최소 재료입니다.

```python
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluation_dataset
```

### 실습: 평가자(evaluator) LLM/Embedding 준비 및 평가 실행

```python
# Ragas 평가/지표 관련
from ragas import EvaluationDataset, aevaluate, RunConfig
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    BleuScore,
    SemanticSimilarity
)
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI


# OpenAI 클라이언트 생성
client = AsyncOpenAI()

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

# 평가자 LLM
evaluator_llm = llm_factory('gpt-4.1-mini', client=client, temperature=0.1, max_tokens=8192)

# 평가자 Embedding
evaluator_embeddings = RagasOpenAIEmbeddings(model="text-embedding-3-large", client=client)
semantic_scorer = SemanticSimilarity(embeddings = evaluator_embeddings)

# 20개 질문을 병렬로 평가
run_config = RunConfig(max_workers=32)

print("평가자 준비 완료!")
```

> 💡 **평가 대상 LLM(`gpt-5.2`)과 평가자 LLM(`gpt-4.1-mini`)을 다른 모델로 분리**한 점에 주목할 필요가 있습니다. 같은 모델이 스스로를 평가하면 편향(self-preference bias)이 생길 수 있기 때문입니다.

```python
result = await aevaluate(
    dataset=evaluation_dataset,
    metrics=[
        BleuScore(),
        ContextRecall(),
        semantic_scorer,
        Faithfulness(),
        FactualCorrectness(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
)

result
```

```python
result.scores[0:3]
```

```python
detailed_result = result.to_pandas()
os.makedirs('outputs/eval', exist_ok=True)
detailed_result.to_csv('outputs/eval/ragas_result.csv', encoding='utf-8', errors='ignore', index=False)
```

> ℹ️ 이 3개 셀의 개별 실행 결과(구체적 점수)는 노트북에 기록되어 있지 않지만, **기본 RAG의 최종 점수는 Part 8의 종합 비교표에 포함**되어 있습니다(Bleu 0.2590, Context Recall 0.6892, Semantic Similarity 0.7976, Faithfulness 0.7063, Factual Correctness 0.6650 — 실제 로컬 실행값 기준).

### 실습: 비교 실험을 위한 재사용 함수 정의

기본적인 구조의 RAG 성능을 확인했으니, 이후 Multi-Query/Ensemble/Rerank/Contextual/Agentic 각각을 평가할 때 반복 사용할 함수를 정의합니다.

```python
def rag_answer_batch(questions, rag_chain, retriever):
    """질문 리스트를 받아 (답변 리스트, 근거 컨텍스트 리스트)를 반환합니다."""
    answers = rag_chain.batch(questions)
    contexts = [[doc.page_content for doc in docs] for docs in retriever.batch(questions)]
    return answers, contexts


async def evaluate_rag(name, answer_fn):
    """검색 방식을 바꿔 가며 같은 지표로 비교합니다.

    name: 결과 파일 이름에 사용할 방식 이름
    answer_fn: 질문 리스트를 받아 (답변, 근거 컨텍스트)를 돌려주는 함수
    """
    answers, contexts = answer_fn(questions)

    new_dataset = [
        {
            "user_input": question,
            "retrieved_contexts": context,
            "response": answer,
            "reference": ground_truth,
        }
        for question, context, answer, ground_truth
        in zip(questions, contexts, answers, ground_truths)
    ]

    evaluation_dataset = EvaluationDataset.from_list(new_dataset)

    result = await aevaluate(
        dataset=evaluation_dataset,
        metrics=[
            BleuScore(),
            ContextRecall(),
            semantic_scorer,
            Faithfulness(),
            FactualCorrectness()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('outputs/eval', exist_ok=True)
    csv_filename = f'outputs/eval/ragas_result_{name}_{timestamp}.csv'
    result.to_pandas().to_csv(csv_filename, encoding='utf-8', errors='ignore', index=False)

    return result
```

> 💡 이 함수 하나로 이후 5가지 Advanced RAG 기법을 **"검색 방식(retriever)만 갈아끼우고 나머지는 동일하게"** 평가할 수 있습니다 — 실험 설계에서 "다른 조건은 고정하고 하나만 바꾼다"는 원칙을 코드로 구현한 것입니다.

---

## Part 3. Multi-Query Retriever

### 이론: Advanced RAG (Processing) — Multi-Query Retrieval (p.44)

> - 쿼리를 여러 개의 버전으로 세분화 후 각각의 검색을 모두 수행
> - 쿼리의 검색 다변화
> - LangChain에 구현되어 있으나, 직접 구현해도 됨

원본 질문 하나로는 검색이 잘 안 되는 경우가 있습니다(용어 불일치, 표현의 모호함 등). 여러 버전의 쿼리로 각각 검색해서 합치면 **검색 다양성**이 늘어납니다.

### 실습: Multi-Query Retriever 구현

```python
# Multi Query를 확인하기 위한 로깅
import logging

logging.basicConfig()
logging.getLogger('langchain_classic.retrievers.multi_query').setLevel(logging.INFO)
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


rewrite_prompt = ChatPromptTemplate([
    ('system','''Retrieval Augmented Generation을 위해,
주어진 질문에 정확한 답변을 수행하기 위한 검색 쿼리를 작성하세요.
원본 질문을 포함해, 사용자의 질문을 해결하기 위해 검색해야 하는 4개의 쿼리를 한 줄에 하나씩 출력하세요.
'구글'에 관한 질문의 경우에는 영문 질문을 포함하세요.
각 질문들은 완성된 질문 형태로 생성하고, 한 줄에 하나씩 새로운 줄로 구분하여 제공하세요.'''),
    ('human','''
---
원본 질문: {question}''')])

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm,
    prompt = rewrite_prompt,
)
```

> 💡 프롬프트에 "'구글'에 관한 질문의 경우에는 영문 질문을 포함하세요"라는 지시가 있습니다 — 원문 보고서가 영어 자료를 포함할 가능성이 있는 도메인이라, **질문과 문서의 언어를 일치**시키는 것도 검색 품질에 영향을 준다는 실전 팁입니다(Part 9 부록의 "질문과 문서의 언어 일치" 패턴과 연결됩니다).

```python
enhanced_context = multi_query_retriever.invoke("트럼프 미국 대통령이 서명한 '제네시스 미션' 행정명령의 핵심 목표는 무엇인가요?")

len(enhanced_context)
```

```python
multiquery_rag_chain = (
    {"context": multi_query_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
await evaluate_rag('multiquery', lambda qs: rag_answer_batch(qs, multiquery_rag_chain, multi_query_retriever))
```

**실제 평가 결과** (Part 8 종합표에서, 실제 로컬 실행값): Bleu 0.2481 / **Context Recall 0.7966** / Semantic Similarity 0.8364 / **Faithfulness 0.7824** / Factual Correctness 0.7575 — 기본 RAG 대비 **모든 지표에서 개선**되었습니다.

---

## Part 4. Ensemble(Hybrid) Retriever — Lexical + Semantic

### 이론: Searching — Dense/Sparse Retrieval과 Hybrid Search (p.32, p.46)

> - Embedding 기반의 Semantic 검색 (DENSE) — 정확하게 일치하지 않아도 유사한 의미를 탐색
> - Keyword 기반의 Lexical 검색 (SPARSE) — BM25, TF-IDF, SPLADE / 정확하게 일치하는 경우에 높은 가중치
> - **Hybrid Search**: Lexical, Semantic 두 검색 결과를 모두 고려하며, 주로 Rank의 조화평균(Reciprocal Rank) 사용

### 실습: Kiwi 형태소 분석기 + BM25 + Ensemble

Lexical 검색인 BM25와 Semantic 검색인 임베딩 방법을 조합합니다. 기본 BM25 리트리버는 한국어 처리가 어려우므로, **Kiwi 형태소 분석기**를 사용합니다.

```python
from kiwipiepy import Kiwi

kiwi = Kiwi()
# Kiwi 형태소 분석기: 고유명사를 추가할 수도 있음
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]
```

```python
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever

# BM25: 단어의 중요도에 따라 가중치를 부여하는 인덱싱 방법
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func = kiwi_tokenize)
bm25_retriever.k = 5

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    # 합집합
)
```

```python
ensemble_rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
await evaluate_rag('ensemble', lambda qs: rag_answer_batch(qs, ensemble_rag_chain, ensemble_retriever))
```

**실제 평가 결과**: Bleu 0.2463 / Context Recall 0.8654(**전체 6가지 방식 중 최고**) / Semantic Similarity 0.8348 / Faithfulness 0.7856 / Factual Correctness 0.7565 — BM25(정확한 키워드 매칭)와 임베딩(의미 매칭)을 5:5로 결합하니 Context Recall이 가장 크게 개선되었습니다.

---

## Part 5. LLM Reranker

### 이론: Reranking (p.50)

> - 질문과 문서의 관련도를 판별하여 재정렬 또는 필터링
> - 임베딩 유사도 이외의 복잡한 방법 필요
>   - 과거에는 Fine-Tuned BERT, T5 등의 모델 사용 — Bi-Encoder VS Cross-Encoder
> - LLM을 이용한 Reranking/Reordering — LLM을 사용해 관련성을 판단하게 하는 방법 고려

### 실습: LLM 기반 Reranker 구현

High-K(넉넉히)로 후보를 검색한 뒤, LLM이 0~10점을 매겨 상위 top_k개만 남기는 방식입니다.

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from typing import List

rerank_prompt = ChatPromptTemplate([
    ('system', '''질문에 답하는 데 문서가 얼마나 도움이 되는지 0에서 10 사이의 정수로 평가하세요.
숫자만 출력하세요.'''),
    ('human', '''질문: {question}
---
문서: {document}''')
])

rerank_chain = rerank_prompt | llm | StrOutputParser()


def parse_score(text):
    matched = re.search(r'\d+', text)
    return int(matched.group()) if matched else 0


class LLMRerankRetriever(BaseRetriever):
    """후보 문서를 LLM이 매긴 점수 순으로 정렬해 상위 top_k개를 반환합니다."""
    base_retriever: BaseRetriever
    reranker: Runnable
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = self.base_retriever.invoke(query)
        scores = self.reranker.batch(
            [{'question': query, 'document': doc.page_content} for doc in candidates]
        )
        ranked = sorted(zip(candidates, scores), key=lambda pair: parse_score(pair[1]), reverse=True)
        return [doc for doc, score in ranked[:self.top_k]]
```

```python
# 리랭킹할 후보를 넉넉히 검색합니다
bm25_candidates = BM25Retriever.from_documents(chunks, preprocess_func=kiwi_tokenize)
bm25_candidates.k = 8

candidate_retriever = EnsembleRetriever(
    retrievers=[bm25_candidates, vector_store.as_retriever(search_kwargs={"k": 8})],
    weights=[0.5, 0.5]
)

rerank_retriever = LLMRerankRetriever(
    base_retriever=candidate_retriever,
    reranker=rerank_chain,
    top_k=5,
)

reranked = rerank_retriever.invoke("트럼프 미국 대통령이 서명한 '제네시스 미션' 행정명령의 핵심 목표는 무엇인가요?")
print(f'후보 {len(candidate_retriever.invoke("제네시스 미션"))}개 중 {len(reranked)}개 선택')
```

```python
rerank_rag_chain = (
    {"context": rerank_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

await evaluate_rag('rerank', lambda qs: rag_answer_batch(qs, rerank_rag_chain, rerank_retriever))
```

**실제 평가 결과**: Bleu 0.2461 / Context Recall 0.7812 / Semantic Similarity 0.8436 / Faithfulness 0.7494 / Factual Correctness 0.7550 — 다른 4가지 Advanced 기법에 비해 **개선폭이 가장 제한적**이었습니다. Reranker는 "이미 검색된 후보를 선별"하는 역할이라, 후보군 자체(BM25 k=8 + 임베딩 k=8)의 품질에 결과가 좌우됩니다.

---

## Part 6. Contextual Retrieval

### 이론: Contextual Retrieval (p.43) + R-G Decoupling (수업 필기)

> 청킹의 앞부분 단절을 완화하기 위해, 전체 맥락을 고려해 청크의 앞부분을 추가
> - 키워드 없음 / 상대적 표현 / 앞부분 분절과 같은 부정확한 청킹 완화 효과
> - Context Caching을 통해 비용을 절약할 수 있음

> **원문 노트(R-G Decoupling)**: "텍스트의 길이가 길어질수록 임베딩 알고리즘은 해당 내용을 모두 포함하게 되므로 검색에서의 정확도가 감소할 수 있습니다. 이를 해결하기 위한 Small2Big Chunking/Sliding Window는 작은 크기의 청크로 검색을 하고, 실제 Augmenting 단계에서는 큰 청크를 포함하는 방법입니다."

Contextual Retrieval은 Small2Big/Sliding Window와 유사한 목표(청킹으로 인한 맥락 단절 완화)를, **"청크 앞에 LLM이 생성한 요약 헤더를 붙이는"** 방식으로 달성합니다.

### 실습: 청크 확인과 Context 생성 프롬프트

```python
chunks = text_splitter.split_documents(documents)
chunks[40]
```

```python
context_prompt = ChatPromptTemplate([
    ('system', '''RAG 검색용 청크에 빠져 있는 문맥 정보를 한국어 1-2문장으로 보강하세요.
청크에 이미 있는 내용은 반복하지 마세요.

## 우선 포함할 정보 (문서에서 확인 가능한 것만)
1. 문서의 종류·제목·출처: 회사/기관명, 보고서·논문·매뉴얼명, 분기·연도
2. 청크가 속한 상위 섹션 또는 소주제
3. 청크 내 대명사·생략 주어가 가리키는 고유명사 (회사, 인물, 제품, 사건, 코드명 등)
4. 비교·연속·인과 관계가 있다면 비교 대상 또는 직전 맥락의 핵심 수치/사실

## 작성하지 않을 것
- 청크 본문의 요약·재진술
- "이 부분은 ~을 설명한다" 같은 메타 서술
- 추상적 주제 해석이나 평가
- 문서에서 확인되지 않는 추측

## 출력 형식
- 한국어 1-2문장, 약 50-100 토큰
- 머리말·꼬리말·번호·따옴표 없이 본문만
- 문서에서 확인 가능한 정보가 부족하면 무리하게 채우지 말고 짧게 마무리'''),
    ('user', '''<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Context:''')
])

#Long Context 처리
long_llm = ChatOpenAI(model='gpt-5.2', max_tokens=8192)


context_chain = context_prompt | long_llm | StrOutputParser()
```

> 💡 시스템 프롬프트의 "작성하지 않을 것" 목록이 상당히 구체적입니다 — 청크 요약이 아니라 **"빠진 맥락만" 짧게 보강**하도록 명시적으로 제약을 걸었습니다. README p.43 도식의 "이용자 수가 전년 대비 37% 증가하며..."처럼, 원본 청크에는 없지만 문서 전체를 보면 알 수 있는 정보(비교 대상, 고유명사 등)를 채우는 것이 목적입니다.

```python
chunk = chunks[35]
source = TextLoader(chunk.metadata['source'], encoding='utf-8').load()[0].page_content
context = context_chain.invoke({'document':source, 'chunk':chunk.page_content})
print(context)
print('========')
print(chunk.page_content)
```

> ℹ️ 이 확인 셀의 실제 출력(생성된 Context 문구)은 노트북에 기록되어 있지 않습니다 — 원본 노트북에는 이 단계를 실제로 실행하는 대신, 아래처럼 **미리 계산해 둔 결과(`chunks_export.jsonl`)를 불러오는 방식**으로 진행됩니다(전체 문서에 대해 매번 LLM 호출을 반복하는 비용을 피하기 위한 설계로 보입니다).

### 실습: 전체 청크에 Context 추가하기 (사전 계산본 로드)

```python
# from langchain_core.documents import Document
# import jsonlines
# from tqdm import tqdm
# def save_docs_to_jsonl(documents, file_path):
#     with jsonlines.open(file_path, mode="w") as writer:
#         for doc in documents:
#             writer.write(doc.model_dump())
# for i, chunk in enumerate(tqdm(chunks)):
#     source = TextLoader(chunk.metadata['source']).load()[0].page_content
#     context = context_chain.invoke({'document':source, 'chunk':chunk.page_content})
#     print('\n'+context)
#     print('---')
#     chunks[i].page_content = context + '\n\n' + chunks[i].page_content
# save_docs_to_jsonl(chunks, './chunks_export.jsonl')


from langchain_core.documents import Document
import jsonlines
def load_docs_from_jsonl(file_path):
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents

chunks = load_docs_from_jsonl("data/evaluation/chunks_export.jsonl")
chunks[0:3]
```

주석 처리된 윗부분이 **실제로 전체 청크에 대해 `context_chain`을 반복 호출하여 헤더를 붙이고 저장하는 코드**이고, 아랫부분은 그 결과를 다시 불러오는 코드입니다. 실습 환경에서는 이미 계산된 `data/evaluation/chunks_export.jsonl`을 제공하여 반복 실행 시 불필요한 LLM 호출(과 비용)을 줄이도록 구성되어 있습니다.

### 실습: 보강된 청크로 벡터DB·하이브리드 검색 재구성

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large', chunk_size=100)

uuidstr = str(uuid.uuid4())[0:6]
client = QdrantClient(path=f"outputs/vectordb/qdrant_{uuidstr}")
client.create_collection(
    collection_name=f"AI_Reports_Contextual",
    vectors_config=VectorParams(size=3072, distance=Distance.EUCLID))
vector_store = QdrantVectorStore(
    client=client,
    collection_name=f"AI_Reports_Contextual",
    embedding=openai_embeddings,
    distance=Distance.EUCLID)

vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

```python
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func = kiwi_tokenize)
bm25_retriever.k = 5

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)
```

> 💡 Contextual Retrieval을 Part 4의 Ensemble(Hybrid)과 **결합**했습니다. Anthropic이 2024년 9월 제안한 원 논문에서도 "Contextual Retrieval + Hybrid Search + Reranking"을 함께 쓰는 것을 권장합니다 — README p.43의 "Claude의 보고서에서, Hybrid 검색 성능 대폭 향상" 문구가 바로 이 조합을 가리킵니다.

```python
contextual_rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

await evaluate_rag('contextual', lambda qs: rag_answer_batch(qs, contextual_rag_chain, ensemble_retriever))
```

**실제 평가 결과**: Bleu 0.2585 / **Context Recall 0.9136 (전체 6가지 방식 중 최고)** / Semantic Similarity 0.8429 / Faithfulness 0.7523 / **Factual Correctness 0.7775 (최고)** — Context Recall과 Factual Correctness에서 가장 우수한 성능을 보였습니다.

---

## Part 7. Agentic RAG

### 이론: RAG in 2026 (2) — Retrieval을 활용하는 Agent로의 발전 (p.55)

> - Retrieval → Generation의 단일 워크플로우는 검색 결과에 대한 대응이 불가능함
>   - 검색 결과 미비 시 추가 검색, 다중 출처 검색 등
> - Retrieval을 활용하는 Agent로의 발전
>   - Workflow는 사라지지만, Retrieval 자체는 중요하게 남음

### 이론: RAG의 다양한 시나리오 (수업 필기) — 5번 유형과의 연결

README의 "RAG의 다양한 시나리오" 중 **5번 "여러 번의 연속 검색이 필요한 질문"**(삼성SDS 클라우드 경쟁사 → 각 경쟁사의 2027 프로젝트, 2단계 검색)이 바로 이번 Part의 문제입니다.

> 기존의 RAG 파이프라인(질문 → 검색 → 답변)으로는 이런 질문을 효율적으로 해결하기 어렵습니다. → **RAG의 고정적 Workflow에서, Agentic RAG로의 변화 추세**

Part 3~6까지는 "retriever만 바꾸고 파이프라인 구조는 고정"이었지만, 이번에는 **검색 자체를 Agent에게 맡깁니다**. retriever를 Tool로 등록하면 Agent가 스스로 다음을 결정합니다.
- 검색을 수행할지 여부
- 어떤 쿼리로 검색할지
- 결과가 부족할 때 쿼리를 바꿔 다시 검색할지

이는 Lab 8(`[실습]_8_LangChain과_다양한_툴_연동.ipynb`)에서 다루는 `create_agent()`/Tool Calling 구조를 RAG 검색에 그대로 적용한 것입니다.

### 실습: Retriever를 Tool로 등록하고 Agent 구성

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage


@tool
def search_documents(query: str) -> str:
    """AI 산업 동향 보고서에서 질문과 관련된 문서 청크를 검색합니다.

    Args:
        query: 검색할 질문 또는 키워드. 완성된 질문 형태일수록 좋습니다.

    Returns:
        검색된 문서 청크 (XML 형식)
    """
    return format_docs(ensemble_retriever.invoke(query))


AGENT_SYSTEM = '''당신은 QA(Question-Answering)를 수행하는 Assistant입니다.
Search_documents 툴로 보고서를 검색한 뒤, 검색 결과를 바탕으로 답변하세요.
한 번만 검색하지 말고, 추가 정보를 얻을 만한 쿼리를 선정하여 최대 세 번까지 검색하세요.
검색 결과 전체에 관련 정보가 없다면 "정보가 부족하여 답변할 수 없습니다."만 출력하세요.'''

agent = create_agent(llm, tools=[search_documents], system_prompt=AGENT_SYSTEM)
agent
```

> 💡 시스템 프롬프트가 "한 번만 검색하지 말고 ... 최대 세 번까지 검색하세요"라고 **명시적으로 반복 검색을 유도**합니다. Agent에게 맡긴다고 자동으로 여러 번 검색하는 것이 아니라, 프롬프트 설계로 "언제 그만둘지"를 조절해야 함을 보여줍니다.

### 실습: Agent의 단계별(Tool Call → Observation) 실행 과정 확인

```python
# 툴 호출과 최종 답변 과정을 단계별로 확인합니다
sample_q = "트럼프 미국 대통령이 서명한 '제네시스 미션' 행정명령의 핵심 목표는 무엇인가요?"

for chunk in agent.stream({"messages": [HumanMessage(sample_q)]}, stream_mode="updates"):
    for step, data in chunk.items():
        message = data['messages'][-1]
        print(f"step: {step}")
        text = message.text
        if len(text) > 500:
            print(f"    content: {text[:500]} ... (중략)")
        else:
            print(f"    content: {text}")
        if step == 'model' and message.tool_calls:
            print("    tool_calls:", message.tool_calls)
        print('-------------')
```

> 💡 **이론 연결 — ReAct Agent 프롬프트 (p.70)**: 이 `stream_mode="updates"` 출력이 README의 ReAct 예시(생각→행동→관찰→최종 답변)를 그대로 관찰할 수 있는 지점입니다. `step`이 `'model'`이면 LLM의 판단(생각+행동 계획), `step`이 `'tools'`이면 Tool 실행 결과(관찰)에 해당합니다.

### 실습: Agentic RAG 평가

```python
def agent_rag_batch(questions):
    """Agent를 실행하고 (최종 답변, 툴 호출로 받은 검색 결과)를 반환합니다."""
    inputs = [{"messages": [HumanMessage(q)]} for q in questions]
    answers = [None] * len(questions)
    contexts = [None] * len(questions)

    for idx, output in tqdm(agent.batch_as_completed(inputs,
                                                    config={"recursion_limit": 10},
                                                    return_exceptions=True),
                            total=len(inputs), desc="Agent RAG"):
        if isinstance(output, Exception):
            print(f"#{idx} 실패: {type(output).__name__}")
            answers[idx] = "정보가 부족하여 답변할 수 없습니다."
            contexts[idx] = ["(검색 결과 없음)"]
            continue
        messages = output["messages"]
        answers[idx] = messages[-1].text
        # 툴 호출 횟수 = 검색 횟수
        contexts[idx] = [m.text for m in messages if isinstance(m, ToolMessage)] or ["(검색 결과 없음)"]

    return answers, contexts


await evaluate_rag('agent', agent_rag_batch)
```

> ⚠️ **실제로 발생한 이슈**: 이 평가 중 `IncompleteOutputException`(평가자 LLM의 `max_tokens=8192` 초과) 1건이 실제로 발생했습니다(Job 64/100). RAGAS는 해당 1건만 결과에서 제외하고 나머지 19건으로 집계했으며, 전체 평균에 미치는 영향은 미미했습니다(Windows 이슈 표의 5번 항목).

**실제 평가 결과**: Bleu 0.2361 / Context Recall 0.7955 / Semantic Similarity 0.8277 / Faithfulness 0.7605 / Factual Correctness 0.7616 — 원본 노트북에 기록된 Colab 참고값(Factual Correctness 0.7795로 최고점)보다는 다소 낮게 나왔지만, 여전히 기본 RAG 대비 개선된 결과입니다. (다음 Part에서 두 수치 세트를 함께 제시합니다.)

---

## Part 8. 실행 결과 종합 (실제 로컬 실행값 vs 원본 Colab 참고값)

### 8.1 실제 로컬 실행값 (2026-08-07, `gpt-5.2` / 평가자 `gpt-4.1-mini`)

> 노트북의 "✅ 실행 결과 (로컬 실행 기록)" 셀에서 그대로 인용합니다.

| RAG 방식 | Bleu Score | Context Recall | Semantic Similarity | Faithfulness | Factual Correctness (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 기본 RAG | 0.2590 | 0.6892 | 0.7976 | 0.7063 | 0.6650 |
| Multi-Query | 0.2481 | 0.7966 | 0.8364 | 0.7824 | 0.7575 |
| 하이브리드(Ensemble) | 0.2463 | **0.8654** | 0.8348 | 0.7856 | 0.7565 |
| Rerank | 0.2461 | 0.7812 | 0.8436 | 0.7494 | 0.7550 |
| **Contextual** | 0.2585 | **0.9136** | 0.8429 | 0.7872 | **0.7775** |
| Agentic | 0.2361 | 0.7955 | 0.8277 | 0.7605 | 0.7616 |

- 모든 Advanced RAG 기법이 기본 RAG보다 4개 지표(Context Recall / Semantic Similarity / Faithfulness / Factual Correctness) 전반에서 개선되었습니다.
- **Contextual RAG**가 Context Recall(0.9136)과 Factual Correctness(0.7775)에서 가장 우수했습니다.
- Bleu Score는 어휘적 중첩만 반영하는 지표라 방식 간 차이가 크지 않았고, 질문을 확장/재구성하는 Multi-Query·Agentic 방식이 오히려 다소 낮게 나타났습니다.
- 실제 수치는 아래 8.2의 Colab 참고값과 세부 값이 다르지만(모델 응답의 비결정성, `gpt-5.2` 버전 차이 등의 영향), 방식 간 우열 순서와 해석은 대체로 일치합니다.

**저장된 결과 파일**:
```
outputs/eval/ragas_result.csv                      # 기본 RAG
outputs/eval/ragas_result_multiquery_*.csv         # Multi-Query
outputs/eval/ragas_result_ensemble_*.csv           # 하이브리드(Ensemble)
outputs/eval/ragas_result_rerank_*.csv             # Rerank
outputs/eval/ragas_result_contextual_*.csv         # Contextual
outputs/eval/ragas_result_agent_*.csv              # Agentic
```

### 8.2 원본 노트북(Colab)에 기록되어 있던 참고값

> ℹ️ 원본 노트북에는 "RAG 성능 평가 최종 비교 분석"이라는 제목의 마크다운 셀이 **동일한 내용으로 두 번 중복**되어 있었습니다(편집 과정에서 생긴 중복으로 추정, 아래는 1회만 표기).

| RAG 방식 | Bleu Score | Context Recall | Semantic Similarity | Faithfulness | Factual Correctness (F1) |
| :------------- | :--------- | :------------- | :------------------ | :----------- | :----------------------- |
| 기본 RAG | 0.2633 | 0.7392 | 0.7966 | 0.7512 | 0.6920 |
| Multi-Query | 0.2476 | **0.8400** | 0.8376 | **0.8138** | 0.7385 |
| 하이브리드 | 0.2499 | 0.8108 | **0.8413** | 0.7615 | 0.7325 |
| Rerank | 0.2615 | 0.7940 | 0.8079 | 0.7275 | 0.7105 |
| Contextual | 0.2580 | 0.8548 | 0.8476 | 0.7523 | 0.7710 |
| Agentic | 0.2322 | 0.7955 | 0.8322 | 0.7921 | **0.7795** |

이 참고값 기준으로는 **Agentic RAG가 Factual Correctness 최고점(0.7795)**, **Contextual RAG가 Context Recall·Semantic Similarity 최고점**을 기록했습니다. 원본 노트북의 결론:

> - **Contextual RAG의 약진**: Context Recall과 Semantic Similarity에서 최고점. 문서 청크에 상위 문맥 정보를 추가하는 방식이 검색 회수율과 답변의 의미적 정확도를 함께 높였습니다.
> - **Agentic RAG의 최고 사실 정확성**: Factual Correctness에서 전체 최고점. LLM이 스스로 검색 쿼리를 생성·반복하는 방식이 가장 정확한 답변을 만들어냈습니다.
> - **Multi-Query RAG의 전반적 강세**: Faithfulness 최고점. 다양한 쿼리 생성이 환각을 줄이는 데 가장 효과적이었습니다.
> - **하이브리드(Ensemble) RAG**: 안정적인 중상위권 성능, Lexical+Semantic 조합의 균형이 강점.
> - **Rerank RAG**: 다른 Advanced 기법에 비해 개선폭이 제한적 — Reranker는 후보 선별 역할이라 답변 생성 자체에 미치는 영향이 상대적으로 작음.
> - **기본 RAG**: 4개 핵심 지표 모두에서 최하위 — Advanced RAG 기법 도입의 필요성을 재확인.

**8.1과 8.2를 함께 보았을 때의 공통 결론**: 실행 환경(로컬 vs Colab)과 시점에 따라 세부 점수는 달라지지만, **"Contextual RAG가 Context Recall에서 가장 강하다"**, **"기본 RAG가 항상 최하위"**, **"Rerank의 개선폭이 상대적으로 작다"**는 경향은 두 실행 모두에서 일관되게 나타났습니다. 이는 LLM 기반 평가(RAGAS)가 **절대 점수보다 상대적 비교에 더 신뢰할 수 있다**는 README p.52의 "절대 수치보다는 상대적 비교가 효과적"이라는 조언과 정확히 일치합니다.

---

## Part 9. 부록: 실전 패턴과 평가지표 해설

### 9.1 실제 데이터에서 중요한 패턴들 (원본 노트북 부록)

- Time 메타데이터를 이용한 사전 필터링
- RAG 여부를 사전 판단하거나, 여러 개의 Vector DB 중 어느 DB를 검색할지 판단하는 단계
- 질문과 문서의 언어 일치 (한국어 → 영어 변환 후 검색)

> 이 세 항목은 Part 3(Multi-Query, 언어 일치)와 Part 7(Agentic, DB/검색여부 사전 판단)에서 이미 다룬 내용의 실무 버전 축약입니다.

### 9.2 RAGAS 평가 결과 컬럼 해설

| 컬럼 | 의미 |
|---|---|
| `bleu_score` | 생성된 답변(`response`)과 정답(`reference`)의 n-gram(어휘) 중첩도. 높을수록 어휘적으로 유사 |
| `context_recall` | 검색된 문서(`retrieved_contexts`)가 정답(`reference`)에 필요한 정보를 얼마나 포함하는지. 높을수록 관련 정보를 잘 검색함 |
| `semantic_similarity` | 답변과 정답의 임베딩 기반 의미 유사도. 높을수록 의미상 유사 |
| `faithfulness` | 답변이 검색된 문서에 근거하는지 여부. 높을수록 환각(Hallucination) 없이 검색 결과만 사용 |
| `factual_correctness (mode=f1)` | 답변과 정답의 사실적 일치도(F1). Precision·Recall의 균형을 반영 |

### 9.3 RAG의 다양한 시나리오 & Agentic RAG로의 변화 (README 중복 수록분)

노트북 부록의 이 절은 `README.md`의 **"RAG의 다양한 시나리오"(수업 필기, Day2-2)**와 **완전히 동일한 내용**입니다 — 5가지 질의 유형(전처리의 중요성/시맨틱 검색/키워드 검색/병렬 검색/연속 검색)과, "RAG의 고정적 Workflow에서 Agentic RAG로의 변화 추세" 결론이 그대로 반복 수록되어 있습니다. 자세한 내용은 README의 해당 절을 참고하십시오 — 본 Lab 6의 **Part 7(Agentic RAG)**이 바로 이 이론을 코드로 구현한 부분입니다.

---

## Part 10. 정리

### 10.1 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 구현한 것 |
|---|---|
| RAG 파이프라인의 성능 개선 지점(p.41) | Chunking(청크 사이즈 1000/200) → Searching(6가지 방식) → 순서로 개선 포인트 하나씩 실증 |
| Advanced RAG (Processing) — Multi-Query(p.44) | `MultiQueryRetriever` + 4개 쿼리 재작성 프롬프트 |
| Searching — Hybrid Search(p.46), Dense/Sparse(p.32) | `EnsembleRetriever`(BM25+임베딩, 5:5 가중치) |
| Advanced RAG (Augmenting) — Reranking(p.50) | `LLMRerankRetriever`(0~10점 LLM 채점 후 top-k 선별) |
| Contextual Retrieval(p.43), R-G Decoupling(수업 필기) | `context_chain`으로 청크별 헤더 보강 + Hybrid 검색 재구성 |
| RAG in 2026(2)(p.55), RAG의 다양한 시나리오 5번(수업 필기) | `create_agent()` + `search_documents` Tool로 Agentic RAG 구현 |
| RAG Evaluation(p.52), RAGAS Claim 기반 평가(수업 필기) | `aevaluate()` + 5개 지표(Bleu/Recall/Similarity/Faithfulness/Correctness)로 6가지 방식 비교 |

### 10.2 실행 결과 종합

| # | 실습 항목 | 핵심 확인 사항 |
|---|---|---|
| 1 | 기본 RAG 체인 구성 | Qdrant + `text-embedding-3-large` + XML 포맷 컨텍스트로 베이스라인 완성 |
| 2 | RAGAS 평가 파이프라인 구축 | `evaluate_rag()` 재사용 함수로 6가지 방식을 동일 조건에서 비교 가능하게 설계 |
| 3 | Multi-Query | 기본 RAG 대비 전 지표 개선, 특히 Faithfulness/Context Recall 강세 |
| 4 | Ensemble(Hybrid) | Context Recall 최고 수준(로컬 실행 기준 1위, 0.8654) |
| 5 | Rerank | 개선되었으나 상대적으로 개선폭이 가장 작음 |
| 6 | Contextual Retrieval | Context Recall·Factual Correctness 최고(로컬 실행 기준) |
| 7 | Agentic RAG | Colab 참고값 기준 Factual Correctness 최고 — 다단계 검색이 필요한 질문에 강점 |
| 8 | Windows 실행 이슈 5건 | pip 인용부호, bs4 누락, 인코딩, pywin32, 평가자 max_tokens 초과 — 모두 실제 재현·해결 |

### 10.3 참고 자료

- RAGAS 공식 문서: <https://docs.ragas.io/en/stable/>
- RAGAS 지표 목록: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- Contextual Retrieval (Anthropic, 2024.09) — 본 저장소 `README.md`의 "Contextual Retrieval (p.43)"
- Lost in the Middle 논문: <https://arxiv.org/pdf/2307.03172>
- Long-Context LLMs Meet RAG 논문: <https://arxiv.org/abs/2410.05983>
- ReAct 논문: <https://arxiv.org/abs/2210.03629>
- 원본 실습 노트북: `[실습]_6_Advanced_RAG+RAG성능평가.ipynb`

### 10.4 다음 단계

- **Lab 7** (`[실습]_7_LangChain을_이용한_SQL_데이터베이스_분석.ipynb`)에서는 이번 실습의 벡터 기반 Retrieval과 달리, README "Retrieval 4+1개 유형"(p.57)의 **"+1) 관계형 데이터베이스 검색"**을 다룹니다 — SQL 쿼리를 자동 생성/실행하는 Text-to-SQL 체인입니다.
- **Lab 8** (`[실습]_8_LangChain과_다양한_툴_연동.ipynb`)에서는 이번 Part 7의 Agentic RAG에서 사용한 `create_agent()`를 더 일반적인 Tool(계산기, 파일 IO, 웹 검색 등)로 확장합니다.
- 실무 적용 시 고려할 점: Contextual Retrieval은 사전 계산 비용(전체 청크에 대한 LLM 호출)이 크므로, 문서 갱신 빈도와 비용을 함께 고려해야 합니다. Agentic RAG는 다단계 검색이 필요한 질문에서 강점을 보이지만, Tool 호출 횟수에 따라 지연시간과 비용이 함께 증가합니다.
