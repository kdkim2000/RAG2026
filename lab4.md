# Lab 4. 벡터 데이터베이스 기반 RAG 어플리케이션 만들기

> 이 실습 교재는 `README.md`(교재_0803.pdf 이론 요약)의 **"3. 벡터스토어와 RAG"(p.20~36)** 이론을 `[실습]_4_벡터_데이터베이스_기반_RAG_어플리케이션_만들기.ipynb`의 코드와 결합하여 재구성한 것입니다. README 부록 "실습(Lab) 목록"의 **Lab #4 "벡터데이터베이스 기반 RAG 어플리케이션"(p.37, Day1-3)**에 해당하며, 소개 문구는 다음과 같습니다.
>
> - `WebBaseLoader()`를 통한 웹페이지 내용 기반의 RAG
> - ChromaDB를 이용하여 뉴스 기사 청크 저장 및 검색

> ⚠️ **실행 상태 안내**: 이 노트북의 73개 셀 전체를 확인한 결과, **어떤 코드 셀에도 `execution_count`나 출력(output)이 기록되어 있지 않습니다.** 즉 Lab 5~8과 달리, 이 노트북은 **코드만 작성되고 아직 한 번도 실행되지 않은(또는 저장 시 출력이 모두 지워진) 상태**로 보존되어 있습니다. 이 교재는 "실행 결과를 임의로 지어내지 않는다"는 원칙을 지키기 위해, 실제 출력값을 인용하는 대신 **각 코드가 무엇을 하는지, 실행하면 어떤 결과가 나올 것으로 기대되는지**를 이론과 코드 구조에 근거하여 설명합니다. 필요하면 이 노트북을 실제로 실행하여 별도로 실행 결과를 기록하는 후속 작업을 진행할 수 있습니다.

## 학습 목표

- 웹 문서(네이버 뉴스 기사)를 API로 수집하고, `WebBaseLoader`로 크롤링·전처리·청킹하는 **전체 Indexing 파이프라인**을 구현한다.
- **OpenAI 임베딩(API 기반)**과 **오픈소스 임베딩(로컬/GPU 기반, Qwen3-Embedding)** 두 가지 방식을 비교하며, "폐쇄망/온프레미스 환경"이라는 실무 제약 조건과 임베딩 모델 선택 기준(파라미터 수·Max Tokens·차원)을 이해한다.
- ChromaDB에 **영구 저장(persist_directory)**하고, API 토큰 제한에 대응하는 **배치 삽입 패턴**을 학습한다.
- `RunnableParallel().assign()`으로 **답변과 근거(context)를 함께 반환**하는 체인을 구성한다.
- 동일한 질문에 대해 OpenAI 임베딩 기반 검색과 오픈 임베딩 기반 검색의 **검색 결과·최종 답변을 나란히 비교**하는 실험을 설계한다.

## Part 0. 개요와 노트북 구조

**노트북 73개 셀의 흐름**:
1. 환경설정(GPU 필요), 라이브러리 설치
2. LLM(`gpt-5.1`)과 **두 종류의 임베딩 모델**(OpenAI `text-embedding-3-large` / 오픈소스 `Qwen3-Embedding-0.6B`) 준비
3. **데이터 준비**: 네이버 검색 API로 뉴스 링크 수집 → `WebBaseLoader`로 본문 크롤링 → 노이즈 텍스트 전처리 → jsonl로 저장
4. **Chunking**: `RecursiveCharacterTextSplitter`로 청크 분할
5. **Vector DB 구성**: Chroma에 OpenAI 임베딩으로 저장(배치 삽입)
6. Retriever 구성, 검색 결과를 XML로 포맷팅
7. Prompt + RAG Chain 구성, 질의응답 테스트
8. `.assign()`으로 근거 포함 답변 체인 구성
9. **오픈 임베딩으로 별도 Chroma DB 구성** → 동일 질문에 대해 OpenAI vs 오픈 임베딩의 검색 결과·최종 답변 비교

---

## Part 1. 환경 준비: LLM과 두 종류의 임베딩 모델

### 이론: LLM 모델의 한계 (p.21)와 RAG (p.22)

> - 높은 수준이 항상 정확성을 의미하지는 않음 — 최신 데이터 / 내부 데이터 / 도메인 특화 데이터에 대한 질문이라면? 정확한 답변을 하지 못하는 Hallucination(환각)의 위험이 큼
> - **검색증강생성(RAG)**: LLM과 정보 검색의 결합 — 질의와 관련된 정보를 검색하여 프롬프트에 함께 전달

이번 실습의 첫 질문("도메인 특화 언어 모델이란 무엇입니까?")은 의도적으로 **일반 LLM이 정확히 답하기 어려운, 최신/특수 도메인 성격의 질문**입니다 — README p.21의 "도메인 특화 데이터에 대한 질문"의 실제 사례입니다.

### 실습: 환경설정과 GPU 요구사항

> ### <필수> 실습을 진행하기 전, GPU를 T4로 설정해 주세요!

이 실습은 **오픈소스 임베딩 모델을 GPU로 직접 구동**하기 때문에 Colab의 T4 GPU 런타임이 필요합니다(OpenAI 임베딩만 사용한다면 GPU가 필요 없지만, 이 실습은 두 방식을 비교하는 것이 핵심입니다).

```python
%pip install dotenv langchain_huggingface transformers sentence_transformers jsonlines langchain langchain-openai langchain-community langchain_chroma -q
```

> 코랩에서 실행 시, Restart 메시지가 나타납니다. 런타임을 재시작하여 패키지를 정리합니다.

`sentence_transformers`(공개 임베딩 모델 실행)와 `langchain_chroma`(Chroma 벡터 DB 연동)가 이번 실습의 핵심 신규 패키지입니다.

```python
import os
from dotenv import load_dotenv
load_dotenv(override=True)

if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.1", reasoning_effort='low')
```

### 실습: OpenAI 임베딩 준비

> OpenAI의 `text-embedding-3-large`는 빠른 속도로 연산이 가능하나, 비용이 발생하며 온라인 모델입니다. 이에 따라, 폐쇄망/온프레미스 환경에서는 공개 임베딩 모델을 사용하여 구현해야 합니다.

```python
from langchain_openai import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
```

### 이론: Embedding 모델의 두 세대 (수업 필기)

> 임베딩 모델도 아래와 같이 세대가 구분됩니다.
> 1. **(~2024) BERT 기반의 모델**: 문맥상 단어/토큰 임베딩의 '평균'으로 문장 임베딩을 구성 — 예: BGE-M3, OpenAI Embedding
> 2. **(2025~) LLM 기반의 모델**: LLM이 이해한 문맥값 자체를 임베딩으로 사용 — 예: **Qwen3 Embedding**

이 노트북이 실제로 사용하는 오픈 임베딩 모델(`Qwen/Qwen3-Embedding-0.6B`)이 바로 README에서 "2025~ LLM 기반 임베딩 모델"의 대표 예시로 언급된 **Qwen3 Embedding**입니다 — 이론과 실습이 정확히 일치하는 지점입니다.

### 실습: 오픈 임베딩 모델 선택 기준 (노트북 원문)

> 오픈 임베딩 모델에서 중요한 파라미터는 다음과 같습니다.
> - **파라미터 수**: 큰 임베딩 모델의 크기는 LLM에 육박합니다. GPU를 고려하여 선택합니다.
> - **Max Tokens**: 임베딩 모델의 최대 토큰보다 큰 데이터를 입력하면, 앞부분만을 이용해 계산하게 되므로 적절한 검색이 되지 않을 수 있습니다.
> - **임베딩 차원**: 큰 차원의 벡터를 생성하는 임베딩 모델은 검색 속도가 감소합니다.

> 현재 한국어 데이터를 임베딩하기 위해 자주 사용하는 모델: **Qwen/Qwen-3-Embedding**(0.6B, 4B, 8B, 32768 토큰 제한) — 알리바바 클라우드의 Qwen 모델을 개량하여 만든 모델. 가장 최신 모델로, BGE-M3와의 성능 비교가 치열합니다.

```python
# 터미널에서 아래 코드를 실행해도 됨
# hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./embedding
```

```python
from sentence_transformers import SentenceTransformer
import torch

# HuggingFace 임베딩 주소 지정하기
# intfloat/multilingual-e5-small , baai/bge-m3, 등의 주소를 입력하여 지정
# GPU에 여유가 있다면 Qwen3 Embedding의 큰 사이즈 (4B, 8B)

model_name = 'Qwen/Qwen3-Embedding-0.6B'
#실제 주소: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

# CPU 설정으로 모델 불러오기
emb_model = SentenceTransformer(model_name, device='cpu',model_kwargs={'torch_dtype':torch.bfloat16})

# 로컬 폴더에 모델 저장하기
emb_model.save('./embedding')

# 모델 메모리에서 삭제
del emb_model
import gc
gc.collect()

print("임베딩 모델 저장 완료!")
```

> 💡 **코드 동작 설명**: HuggingFace Hub에서 `Qwen3-Embedding-0.6B`(6억 파라미터) 모델을 **일단 CPU로 불러와서** 로컬 `./embedding` 폴더에 저장한 뒤, 메모리에서 즉시 삭제(`del` + `gc.collect()`)합니다. 이렇게 "다운로드 후 즉시 언로드"하는 이유는, 바로 다음 셀에서 **GPU로 다시 로드**하기 때문입니다 — 다운로드(디스크 I/O 중심)와 추론(GPU 메모리 중심)의 책임을 분리한 것입니다. 실행하면 모델 가중치(약 1.2GB)가 다운로드되고, 마지막에 `"임베딩 모델 저장 완료!"`가 출력될 것으로 예상됩니다.

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 허깅페이스 포맷의 임베딩 모델 불러오기
open_embeddings = HuggingFaceEmbeddings(model_name= './embedding',
                                  model_kwargs={'device':'cuda',
                                                'model_kwargs':{'torch_dtype':torch.bfloat16}}) # gpu 사용하기

# GPU 로드된 것 확인
print("임베딩 모델 GPU 로드 완료")
```

> 💡 `torch_dtype':torch.bfloat16`으로 **half-precision(16bit)** 연산을 사용해 GPU 메모리 사용량과 추론 속도를 함께 개선합니다 — README p.11의 최신 LLM 운영에서도 흔히 쓰이는 경량화 기법입니다.

### 실습: RAG 적용 전 베이스라인 확인

```python
# Test
llm.invoke("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?")
```

> 💡 **실행하면 기대되는 동작**: RAG를 적용하기 전, **순수 LLM 단독**의 답변을 먼저 확인하는 대조군(baseline) 역할입니다. `gpt-5.1`이 사전학습 지식만으로 일반적인 설명(도메인 특화 파인튜닝 모델의 개념, BloombergGPT·Med-PaLM 같은 일반적으로 알려진 예시)을 내놓을 것으로 예상되며, Part 6에서 **동일한 질문**을 RAG 체인으로 다시 물어 결과를 비교하게 됩니다 — README p.21~22가 강조하는 "RAG 적용 전후 비교"를 그대로 구현한 설계입니다.

---

## Part 2. 데이터 준비: 네이버 뉴스 크롤링 (Indexing 1단계)

### 이론: LangChain Document Loader (p.29)

> 다양한 형식의 파일이나, API 서비스를 연결하여 데이터를 불러오는 구조 — PDF, DOCX, XML 등의 파일 로더 / Arxiv, Tavily, Pubmed 등의 API 로더

### 실습: 네이버 검색 API로 뉴스 링크 수집

```python
import requests
def get_naver_news_links(query, num_links=100):
    """
    query와 num_links를 입력받아 네이버 검색 수행, 네이버 뉴스 URL의 기사만 수집
    """

    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={num_links}&sort=sim"
    # 최대 100개의 결과를 표시
    headers = {
        'X-Naver-Client-Id': '<발급받은 Client ID>',
        'X-Naver-Client-Secret': '<발급받은 Client Secret>'
    }

    response = requests.get(url, headers=headers)
    result = response.json()
    # 특정 링크 형식만 필터링
    filtered_links = []
    for item in result['items']:
        link = item['link']
        if "n.news.naver.com/mnews/article/" in link:
            # 네이버 뉴스 스타일만 모으기
            filtered_links.append(link)

    # 결과 출력
    print(query, ':', len(filtered_links), 'Example:', filtered_links[0])
    # for link in filtered_links:
    #     print(link)

    return filtered_links

filtered_links = []
for topic in ['도메인 특화 언어모델', 'OpenAI', 'GPT', '구글', '가전제품', '넷플릭스']:
    filtered_links += get_naver_news_links(topic, 100)
print('Total Articles:', len(filtered_links))
print('Total Articles(Without Duplicate):',len(list(set(filtered_links))))
filtered_links = list(set(filtered_links))
```

> ⚠️ **보안 참고 — API 키 하드코딩**: 원본 노트북에는 네이버 API `Client-Id`/`Client-Secret`이 코드에 **평문으로 하드코딩**되어 있습니다(위 코드 블록에서는 값을 마스킹했습니다). 실습용 데모에서는 편의상 흔히 이렇게 작성하지만, 실무 코드나 공개 리포지토리에 커밋할 때는 반드시 `.env` 파일(`load_dotenv()`로 로드, 이 노트북 앞부분에서 OpenAI 키에는 이미 이 방식을 쓰고 있습니다)이나 별도 Secret Manager로 분리해야 합니다. 이 노트북은 이미 Git 저장소에 커밋되어 있으므로, 실제 서비스 중인 키라면 **재발급(rotate)을 권장**합니다.
>
> **동작 방식**: 6개 검색어(도메인 특화 언어모델/OpenAI/GPT/구글/가전제품/넷플릭스) 각각에 대해 최대 100개씩 뉴스를 검색하여 최대 600개의 후보 링크를 모으고, 네이버 뉴스 자체 도메인(`n.news.naver.com/mnews/article/`) 형식만 필터링한 뒤 중복을 제거합니다. 여러 주제를 섞은 이유는, 이후 RAG 체인이 검색 범위가 넓은 지식베이스에서도 정확한 청크를 찾아낼 수 있는지 시험하기 위한 것으로 보입니다.

### 실습: `WebBaseLoader`로 뉴스 본문 크롤링

### 이론: RAG의 5-step Process — Indexing (p.25)

```python
# # Jupyter 분산 처리를 위한 설정 (코랩에서는 불필요)
# import nest_asyncio

# nest_asyncio.apply()
```

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

async def get_news_documents(links):
    loader = WebBaseLoader(
        web_paths=links,
        bs_kwargs={'parse_only':bs4.SoupStrainer(class_=("newsct", "newsct-body"))},
                                # newsct, newsct-body만 추출 : 네이버 뉴스 포맷 HTML 요소

        requests_per_second = 10, # 1초에 10개 요청 보내기
        show_progress = True # 진행 상황 출력
    )
    # docs = loader.load() # 기본 코드

    docs = []

    async for doc in loader.alazy_load(): # 순차적 로드 대신 비동기 처리
        docs.append(doc)
    return docs

docs = await get_news_documents(filtered_links)
```

```python
docs[12]
```

> 💡 **코드 동작 설명**: `bs4.SoupStrainer(class_=("newsct", "newsct-body"))`는 BeautifulSoup이 HTML 전체를 파싱하지 않고 **네이버 뉴스 기사 본문 영역(`newsct`, `newsct-body` CSS 클래스)만 골라서 파싱**하도록 제한합니다 — 광고/댓글/헤더 등 불필요한 HTML을 아예 처리하지 않으므로 속도와 정확도를 함께 높입니다. `alazy_load()`(비동기 제너레이터)로 최대 수백 개의 URL을 `requests_per_second=10` 제한 하에 동시에 가져옵니다. `docs[12]`를 실행하면 `Document(page_content='...(기사 본문+주변 UI 텍스트가 섞인 원문)...', metadata={'source': 'https://n.news.naver.com/...'})` 형태의 결과가 출력될 것으로 예상됩니다 — 다음 Part의 전처리가 필요한 이유가 여기서 드러납니다.

### 실습: 전처리 — 크롤링 노이즈 제거

```python
import re

def preprocess(docs):
    noise_texts = [
        '''구독중 구독자 0 응원수 0 더보기''',
        '''쏠쏠정보 0 흥미진진 0 공감백배 0 분석탁월 0 후속강추 0''',
        '''댓글 본문 요약봇 본문 요약봇''',
        '''도움말 자동 추출 기술로 요약된 내용입니다. 요약 기술의 특성상 본문의 주요 내용이 제외될 수 있어, 전체 맥락을 이해하기 위해서는 기사 본문 전체보기를 권장합니다. 닫기''',
        '''텍스트 음성 변환 서비스 사용하기 성별 남성 여성 말하기 속도 느림 보통 빠름''',
        '''이동 통신망을 이용하여 음성을 재생하면 별도의 데이터 통화료가 부과될 수 있습니다. 본문듣기 시작''',
        '''닫기 글자 크기 변경하기 가1단계 작게 가2단계 보통 가3단계 크게 가4단계 아주크게 가5단계 최대크게 SNS 보내기 인쇄하기''',
        'PICK 안내 언론사가 주요기사로선정한 기사입니다. 언론사별 바로가기 닫기',
        '응원 닫기',
        '구독 구독중 구독자 0 응원수 0 ',
    ]

    def clean_text(doc):
        text = doc.page_content
        # 탭과 개행문자를 공백으로 변환
        text = text.replace('\t', ' ').replace('\n', ' ')

        # 연속된 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()

        # 여러 구분자를 한번에 처리
        split_markers = [
            '구독 해지되었습니다.',
            '구독 메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!'
        ]
        for marker in split_markers:
            parts = text.split(marker)
            if len(parts) > 1:
                if marker == '구독 해지되었습니다.':
                    text = parts[1]  # 뒷부분 사용
                else:
                    text = parts[0]  # 앞부분 사용

        # 노이즈 텍스트 제거
        for noise in noise_texts:
            text = text.replace(noise, '')

        # 연속된 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        doc.page_content = text
        return doc

    preprocessed_docs = []
    for doc in docs:

        # 텍스트 정제
        doc= clean_text(doc)
        preprocessed_docs.append(doc)

    return preprocessed_docs

preprocessed_docs = preprocess(docs)
```

```python
preprocessed_docs[2]
```

> 💡 **코드 동작 설명**: 이 `preprocess()` 함수는 네이버 뉴스 페이지의 **UI 문구(구독 버튼, 이모지 반응 카운트, TTS 안내, 글자크기 조절 버튼, AI 요약봇 안내 등)를 정확히 문자열 매칭으로 제거**합니다. 실제 웹 크롤링에서는 이렇게 **사이트별로 반복 등장하는 상투적 UI 텍스트를 하드코딩된 노이즈 목록으로 제거**하는 방식이 매우 흔합니다 — Docling(README p.30)처럼 구조화된 파서를 쓰지 않고 범용 `WebBaseLoader`+BeautifulSoup을 쓸 때 감당해야 하는 비용입니다. `preprocessed_docs[2]`를 실행하면 본문만 깨끗하게 남은 `Document`가 출력될 것으로 예상됩니다.

### 실습: 전처리된 문서를 jsonl로 저장/로드

```python
# 불러온 document 저장하기

import jsonlines
def save_docs_to_jsonl(documents, file_path):
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.model_dump())

# jsonl 파일 불러오기
from langchain_core.documents import Document

def load_docs_from_jsonl(file_path):
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents
```

```python
# 저장
save_docs_to_jsonl(preprocessed_docs, "docs.jsonl")
```

> 💡 크롤링은 시간과 API 호출 비용이 드는 작업이므로, 한 번 수집·전처리한 결과를 `docs.jsonl`로 저장해두면 이후 청킹·임베딩 실험을 반복할 때 **매번 다시 크롤링하지 않고 재사용**할 수 있습니다. (Lab 6에서 `chunks_export.jsonl`을 미리 계산해 재사용한 것과 동일한 패턴입니다.)

---

## Part 3. Chunking: 청크 단위로 나누기 (Indexing 2단계)

### 이론: 적절한 청크 사이즈 선택하기 (p.27)

> - 청크 크기가 작은 경우 — 주변 정보 활용 어려움
> - 청크 크기가 큰 경우 — 불필요한 정보 포함, 임베딩의 정확도 감소
> - 청크 오버랩 — Context의 의미 보존

### 실습: 노트북 원문 — Chunk Size와 검색 결과 Context의 관계

> 전처리가 완료된 docs를 chunk 단위로 분리합니다. `chunk_size`와 `chunk_overlap`을 이용해 청크의 구성 방식을 조절할 수 있습니다.
>
> **Chunk Size * K(검색할 청크의 수) 의 결과가 Context의 길이가 됩니다.**

이 한 문장이 README의 **"Advanced RAG (Augmenting) — Long Context (p.47)"**("Larger K, Bigger Chunk")와 정확히 같은 원리를 실무 공식으로 표현한 것입니다 — 청크 크기와 검색 개수(K) 중 어느 것을 늘려도 LLM에 들어가는 최종 Context 길이가 늘어난다는 뜻입니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 0~1000, 800~1800, 1600~2600, ...
chunks = text_splitter.split_documents(preprocessed_docs)
print(len(chunks))
```

> 💡 이후 Lab 6·8에서도 동일하게 `chunk_size=1000, chunk_overlap=200`(20% 오버랩)을 사용합니다 — 이 저장소의 실습들이 공통된 청킹 기준을 채택하고 있음을 알 수 있습니다. `print(len(chunks))`는 (중복 제거된 뉴스 기사 수 × 기사당 평균 길이 / 1000자 근처의) 청크 개수를 출력할 것으로 예상됩니다.

---

## Part 4. Vector DB 구성하기

### 이론: Searching — 벡터데이터베이스(p.33), 대표적 Vector Database(p.35), Metric Types(p.36)

> - 벡터 형태의 데이터 저장 및 검색을 제공하는 소프트웨어 — 비정형 데이터를 임베딩으로 변환하고, 이를 저장/검색
> - **Milvus, Qdrant, Chroma, Weaviate**: Online/Self Host 사용 지원
> - **Euclidean Distance (L2 Distance)** — 벡터 간의 직선 거리

### 실습: Chroma 벡터 DB 초기화

```python
from langchain_chroma import Chroma

Chroma().delete_collection() # (메모리에 저장하는 경우) 기존 데이터 삭제

# DB 구성하기
db = Chroma(embedding_function=openai_embeddings,
            persist_directory="./chroma_OpenAI",
            # 파일 시스템에 저장 (생략시 메모리에 저장)

            collection_name='Web', # 식별 이름

            collection_metadata={'hnsw:space':'l2'},
            # l2 메트릭 설정(기본값, cosine, mmr 로 변경 가능)
            )
```

> 💡 **이론 연결**: `collection_metadata={'hnsw:space':'l2'}`가 바로 README p.36의 **Euclidean(L2) Distance**를 명시적으로 지정한 것입니다. 주석에 적힌 대로 `cosine`으로 바꾸면 README의 **Cosine Distance**(정규화된 임베딩에 적합)를 쓸 수 있습니다. 이 실습(Lab 4)은 L2, Lab 5(Chroma)는 기본값, Lab 6(Qdrant)은 `Distance.EUCLID`를 사용하여 — 매 실습마다 README의 여러 거리 지표를 골고루 실습해보도록 구성되어 있습니다.
>
> `persist_directory="./chroma_OpenAI"`로 **파일 시스템에 영구 저장**한다는 점도 눈에 띕니다 — Lab 5·6에서는 임시(메모리/세션 한정) Qdrant를 썼지만, 이 실습은 이후 "오픈 임베딩 DB(`./chroma_open`)와 비교"하기 위해 **두 DB를 별도 폴더에 나란히 보존**해야 하므로 영구 저장 방식을 선택한 것으로 보입니다.

### 실습: 배치 삽입 (OpenAI 임베딩 API의 토큰 제한 대응)

> DB에 document를 추가합니다. OpenAI 임베딩은 30만 토큰 동시 처리 제한이 있어, 나눠서 전달합니다.

```python
from tqdm import tqdm
import time
print(len(chunks))
# 300,000 토큰 제한

# 20개씩 추가
for i in tqdm(range(0, len(chunks), 20)):
    db.add_documents(chunks[i:min(i+20, len(chunks))])
    time.sleep(5)
```

> 💡 **참고할 만한 내용 — API 레이트리밋 대응 패턴**: `for i in range(0, len(chunks), 20)`으로 **20개 청크씩 나눠서** `add_documents()`를 호출하고, 매 배치 사이에 `time.sleep(5)`를 넣어 OpenAI Embeddings API의 **분당/요청당 제한(rate limit)과 30만 토큰 동시처리 제한을 피합니다.** 이는 실무에서 대량의 문서를 임베딩할 때 흔히 마주치는 문제(`RateLimitError`, `BadRequestError: too many tokens`)에 대한 표준적인 완화 방법입니다 — 배치 크기(20)와 대기 시간(5초)은 API 요금제(tier)에 따라 조정이 필요합니다.

---

## Part 5. Retriever 구성과 검색 결과 포맷팅

### 이론: RAG의 Retrieval — 검색 (p.24), 벡터DB 활용 예시 (p.34)

> - 데이터베이스 내용 중 질문과 관련 있는 내용이 무엇인지를 알아야 함
> - 질문이 입력되면, 질문의 임베딩과 청크들의 임베딩 유사도 계산 → **Top K Chunks를 Return**

### 실습: Retriever 생성과 검색 테스트

```python
# Top 5 Search(기본값은 4)
retriever = db.as_retriever(search_kwargs={'k':5})
```

```python
context = retriever.invoke("도메인 특화 언어 모델")
context
```

> 💡 **실행하면 기대되는 동작**: "도메인 특화 언어 모델"이라는 질문의 임베딩과 가장 유사한 **상위 5개 뉴스 청크**(`Document` 객체 리스트)가 반환됩니다. 검색어에 특정 기업명이 없으므로, 도메인 특화 LLM을 다룬 IT/AI 관련 기사 청크들이 상위에 올라올 것으로 예상됩니다.

### 실습: 검색 결과를 XML로 직렬화하기

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


print(format_docs(context, metadata_keys=("source", "page", "section")))
```

> 💡 이 `format_docs()` 함수는 Lab 6·7에서도 동일하게 사용되는 이 저장소의 **표준 컨텍스트 포맷터**입니다 — 검색된 문서를 `<document index="N">` 태그로 구분한 XML로 직렬화하여, LLM이 여러 청크의 경계를 혼동하지 않도록 합니다. 이번 실습에서는 `metadata_keys=("source", "page", "section")`을 지정했지만, 실제 `Document.metadata`에는 `WebBaseLoader`가 채운 `source`(원문 URL)만 존재하므로 `page`/`section`은 값이 없어 자동으로 생략(코드의 `if value is None: continue` 로직)될 것으로 예상됩니다.

---

## Part 6. Prompting과 Chain (Augmenting + Generating)

### 이론: RAG 프롬프트 예시 (p.23), RAG의 5-step Process (p.25)

> 정보와 질문을 구분하여 제공 — 정확한 정보의 검색이 무엇보다 중요

### 실습: RAG 프롬프트 작성

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate([
    ("user", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.
---
Context: {context}
---
Question: {question}''')])

prompt.pretty_print()
```

> 이 프롬프트는 Lab 5·6·7의 프롬프트와 같은 구조적 원칙(Context와 Question을 `---`로 명확히 구분, 정보 부족 시 솔직하게 답변 거절)을 공유합니다 — README p.23의 "정보와 질문을 구분하여 제공"이 이 저장소 전체의 공통 프롬프트 패턴임을 보여줍니다.

### 실습: RAG Chain 구성과 질의응답 테스트

> RAG Chain은 프롬프트에 context와 question을 전달해야 합니다. Question을 입력받아, Context를 함께 프롬프트에 전달합니다.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
rag_chain.invoke("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?")
```

> 💡 **실행하면 기대되는 동작**: Part 1에서 LLM 단독으로 물었던 것과 **동일한 질문**입니다. 이번에는 실제로 크롤링된 뉴스 기사 근거가 함께 주어지므로, 답변에 **기사에서 다룬 구체적 도메인 특화 모델명(예: 금융/의료/법률 특화 LLM 사례)**이 포함될 가능성이 높습니다 — RAG 적용 전후 답변의 구체성 차이를 비교하는 것이 이 실습의 핵심 관찰 포인트입니다.

```python
rag_chain.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")
```

> 💡 "관련 링크도 보여주세요"라는 요청은 검색된 청크의 `metadata['source']`(원문 URL)가 `format_docs()`를 통해 프롬프트에 포함되어 있어야 응답할 수 있습니다. 다만 위 `format_docs(context, metadata_keys=("source","page","section"))` 호출과 달리, `rag_chain` 내부의 `format_docs`는 **기본값**(`metadata_keys=("source",)`)으로 호출되므로 `source`(URL)는 포함됩니다 — LLM이 실제 뉴스 URL을 인용하여 답할 것으로 예상됩니다.

```python
rag_chain.invoke("알리바바의 언어 모델 이름은?")
```

> 💡 Part 2에서 수집한 6개 검색어 중 특정 기업명을 직접 포함한 것은 없었으므로, 이 질문에 대한 답은 **간접적으로 언급된 기사**(예: "구글", "OpenAI" 관련 기사에서 경쟁사로서 알리바바/Qwen이 언급되었을 경우)에 의존합니다. 만약 관련 청크가 검색되지 않으면 프롬프트 지시에 따라 `"정보가 부족하여 답변할 수 없습니다."`가 출력될 가능성도 있습니다 — 이는 Part 1 이론(README p.21)의 "Hallucination 없이 모른다고 답하기"가 실제로 테스트되는 지점입니다.

---

## Part 7. `.assign()`으로 근거(Context)까지 함께 반환하기

### 이론: 특수한 Runnables (p.17)

> - `RunnablePassthrough()`: 직전 `Runnable`의 출력을 그대로 Return
> - `RunnableParallel()`: 1개 이상의 `Runnable`을 실행하고, 그 결과를 Dict로 Return
> - `.assign()`: `Runnable` 값을 전달하고, 그 결과를 가져와 결합

### 실습: 근거 포함 답변 체인

> `assign()`을 이용하면, 체인의 결과를 받아 새로운 체인에 전달하고, 그 결과를 가져옵니다.

```python
# assign : 결과를 받아서 새로운 인수 추가하고 원래 결과와 함께 전달

from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    context = retriever | format_docs, question = RunnablePassthrough()).assign(answer=rag_chain_from_docs)

rag_chain_with_source.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")

# retriever가 1번 실행됨
# retriever의 실행 결과를 rag_chain_from_docs 에 넘겨주기 때문에
```

> 💡 **코드 동작 설명**: Part 6의 `rag_chain`은 최종 답변 문자열만 반환하지만, 이 체인은 `{"context": ..., "question": ..., "answer": ...}` **딕셔너리 전체**를 반환합니다. 노트북 주석이 강조하듯 **`retriever`는 딱 1번만 실행**됩니다 — `RunnableParallel(...)`가 만든 `{"context": ..., "question": ...}` 결과를 `.assign(answer=rag_chain_from_docs)`가 그대로 **재사용**하여 `rag_chain_from_docs`(prompt→llm→parser)에 넘기기 때문입니다. 만약 `retriever | format_docs`를 별도로 두 번 호출했다면 검색이 중복 실행되어 비효율적이고, 두 호출 사이에 DB가 갱신되면 결과가 어긋날 수도 있습니다 — `.assign()`의 이런 "한 번 계산한 값을 재사용"하는 특성이 Lab 6·7의 `RunnableParallel().assign()` 체이닝(예: `input→query→result→answer`)에서도 동일하게 활용됩니다.
>
> 이 패턴은 실무적으로 매우 중요합니다 — 최종 사용자에게 답변만 보여주는 것이 아니라 **"어떤 근거로 이 답을 했는지"(context)까지 함께 노출**하면, RAG 시스템의 신뢰성과 검증 가능성(traceability)이 크게 높아집니다.

---

## Part 8. 오픈 임베딩 모델과의 비교 실험

### 이론: Embedding 모델 세대 구분 (수업 필기) + 폐쇄망/온프레미스 제약

이 Part는 Part 1에서 제기된 문제 — "OpenAI 임베딩은 비용이 발생하며 온라인 모델이므로, 폐쇄망/온프레미스 환경에서는 공개 임베딩 모델을 사용해야 한다" — 에 대한 **실제 해결책과 성능 비교**입니다.

### 실습: 오픈 임베딩(Qwen3)으로 별도 DB 구성

> 이번에는 오픈 모델을 사용합니다. 오픈 임베딩을 통해 구성한 DB와 원래 DB를 비교해 보겠습니다.
>
> 이후는 동일합니다.

```python
open_db = Chroma(embedding_function=open_embeddings,
                           persist_directory="./chroma_open", # 별도 폴더에 저장
                           collection_name='Web', # 식별 이름
                           collection_metadata={'hnsw:space':'l2'}
                           )

# 20개씩 추가
for i in tqdm(range(0, len(chunks), 20)):
    open_db.add_documents(chunks[i:min(i+20, len(chunks))])
```

> 💡 **동일한 `chunks`(같은 청크 분할)를 그대로 재사용**하되, 임베딩 함수만 `openai_embeddings` → `open_embeddings`(Qwen3-Embedding-0.6B)로 바꾼 것이 이 실험의 핵심입니다 — 이렇게 **변수를 하나만 바꾸고 나머지 조건은 고정**하는 것이 Lab 6의 `evaluate_rag()` 실험 설계 원칙과 정확히 같습니다. OpenAI 배치 삽입과 달리 이 셀에는 `time.sleep()`이 없는데, 로컬 GPU 추론은 API 레이트리밋이 없기 때문입니다.

```python
open_retriever = open_db.as_retriever(search_kwargs={'k':5})
```

```python
open_retriever.invoke("도메인 특화 언어 모델")
```

```python
rag_chain_open = (
    {"context": open_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### 실습: OpenAI 임베딩 vs 오픈 임베딩 — 검색 결과와 최종 답변 비교

> 예시 질문을 통해 정답과 검색 결과를 비교해 보겠습니다.

```python
questions = ["도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?",
             "인공지능의 최근 발전 방식은? 관련 링크도 보여주세요",
             "알리바바의 언어 모델 이름은?"]
```

```python
# Retriever 비교

contexts_hf = open_retriever.batch(questions)
contexts_openai = retriever.batch(questions)
```

```python
for i in range(len(questions)):
    print(f"-- Question:{questions[i]}")
    oai_chunks = '\n'.join([x.page_content[:50] for x in contexts_openai[i]])
    hf_chunks = '\n'.join([x.page_content[:50] for x in contexts_hf[i]])
    print(f"OpenAI: \n{oai_chunks}")
    print(f"Hf: \n{hf_chunks}")
    print('----------')

```

> 💡 **코드 동작 설명**: 3개 질문 각각에 대해 `.batch()`로 두 Retriever의 검색 결과를 한 번에 가져오고, 각 청크의 **앞 50자만** 나란히 출력합니다. 짧게 자르는 이유는 5개 청크 × 2개 DB × 3개 질문(총 30개 청크)의 미리보기를 한눈에 비교하기 위한 것입니다. 두 임베딩 모델의 "언어 이해 방식"이 다르므로(BERT 세대 평균 풀링 방식의 OpenAI Embedding vs LLM 기반 문맥 이해의 Qwen3 Embedding, README p.265 참고), 같은 질문에도 **상위 5개 청크의 구성이 다를 가능성**이 있습니다.

```python
# 최종 결과 비교
oai_results = rag_chain.batch(questions)
hf_results = rag_chain_open.batch(questions)
```

```python
for i in range(len(questions)):
    print(f"-- Question:{questions[i]}")
    print(f"OpenAI: \n{oai_results[i]}")
    print('--')
    print(f"Hf: \n{hf_results[i]}")
    print('----------')

```

> 💡 **실행하면 기대되는 동작과 관찰 포인트**: 검색된 청크가 달라지면 최종 답변도 달라질 수 있습니다. 특히 3번째 질문("알리바바의 언어 모델 이름은?")은 Part 6에서 예상했듯 관련 정보가 크롤링된 기사에 부족할 가능성이 있으므로, **두 임베딩 모델 모두 "정보가 부족하여 답변할 수 없습니다"로 답하는지, 아니면 한쪽만 관련 청크를 찾아내는지**를 비교하는 것이 이 실습의 실질적 평가 지점입니다. (마지막 셀 `cell-72`는 빈 코드 셀로 남아 있어, 이 비교 결과를 바탕으로 한 추가 분석은 학습자에게 남겨진 것으로 보입니다.)

---

## Part 9. 정리

### 9.1 코드 구성 종합 (실행 결과 대신 — 이 노트북은 미실행 상태)

| # | 실습 항목 | 무엇을 검증하기 위한 코드인가 |
|---|---|---|
| 1 | LLM·임베딩 준비(OpenAI + Qwen3) | 유료 API 임베딩과 오픈소스 GPU 임베딩을 나란히 준비 — "폐쇄망 대응"이라는 실무 시나리오 |
| 2 | 네이버 뉴스 크롤링 + 전처리 | `WebBaseLoader`+`bs4.SoupStrainer`로 특정 사이트의 본문만 추출, 노이즈 텍스트 제거 |
| 3 | Chunking(1000/200) | "Chunk Size × K = Context 길이" 공식을 실제 파라미터로 구현 |
| 4 | Chroma(L2) 배치 삽입 | OpenAI 임베딩 API의 30만 토큰 제한을 배치+대기로 회피 |
| 5 | Retriever + `format_docs` | Top-K 검색 결과를 XML로 직렬화해 LLM에 안전하게 전달 |
| 6 | RAG Chain + 3개 질의 | RAG 적용 전(Part 1)/후(Part 6) 답변 품질 차이를 관찰하도록 설계 |
| 7 | `RunnableParallel().assign()` | retriever 중복 실행 없이 답변+근거를 함께 반환 |
| 8 | 오픈 임베딩 비교 실험 | 동일 청크·동일 질문 조건에서 OpenAI vs Qwen3 임베딩의 검색/답변 차이 비교 |

> 이 표는 "실행하면 어떤 값이 나왔다"가 아니라 **"이 코드가 왜 이렇게 작성되었는가"**를 정리한 것입니다 — 노트북에 실행 결과가 없으므로, 코드의 설계 의도를 정확히 파악하는 것이 이 실습의 실질적 학습 내용입니다.

### 9.2 이론-실습 연결 매핑

| 이론 (README) | 이번 실습에서 구현한 것 |
|---|---|
| LLM 모델의 한계(p.21), RAG(p.22) | RAG 적용 전/후 동일 질문 비교 설계(Part 1 vs Part 6) |
| LangChain Document Loader(p.29) | `WebBaseLoader`+`bs4.SoupStrainer`로 네이버 뉴스 본문만 추출 |
| 적절한 청크 사이즈(p.27), Long Context(p.47) | "Chunk Size × K = Context 길이" 공식과 `chunk_size=1000/overlap=200` |
| Searching — 벡터DB, Vector DB Metric Types(p.33, p.36) | Chroma `persist_directory` 영구 저장 + `hnsw:space: l2`(Euclidean) 명시적 지정 |
| Embedding 모델 세대 구분(수업 필기) | OpenAI(BERT 세대 계열) vs Qwen3-Embedding(LLM 기반 세대) 실제 비교 |
| 특수한 Runnables — `.assign()`(p.17) | `RunnableParallel().assign()`으로 근거 포함 답변 체인 구성 |

### 9.3 참고 자료

- LangChain Document Loaders: <https://python.langchain.com/docs/integrations/document_loaders/>
- Qwen3 Embedding 모델: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- 임베딩 모델 리더보드 (MTEB): <https://huggingface.co/spaces/mteb/leaderboard>
- Chroma: <https://www.trychroma.com/>
- 네이버 검색 API 문서: <https://developers.naver.com/docs/serviceapi/search/news/news.md>
- 원본 실습 노트북: `[실습]_4_벡터_데이터베이스_기반_RAG_어플리케이션_만들기.ipynb`

### 9.4 다음 단계

- **Lab 5** (`[실습]_5_PDF_내용_기반_질의응답_어플리케이션.ipynb`)에서는 웹 문서 대신 PDF를 대상으로 동일한 Indexing→Searching→Augmenting→Generating 파이프라인을 구현하며, "Top-K 검색의 한계"와 요약(Stuff/Map-Reduce/Refine) 전략으로 이어집니다.
- **Lab 6** (`[실습]_6_Advanced_RAG+RAG성능평가.ipynb`)에서는 이번 실습의 단일 Retriever 구조를 Multi-Query/Ensemble(Hybrid)/Rerank/Contextual Retrieval로 고도화하고, RAGAS로 정량 평가합니다 — 이번 실습에서 "육안으로" 비교했던 OpenAI vs 오픈 임베딩 비교(Part 8)를 정량 지표로 재현해볼 수 있습니다.
- **직접 실행해보고 싶다면**: `.env`에 `OPENAI_API_KEY`와 네이버 API 키(`.env`로 분리하여)를 설정하고, Colab에서 GPU(T4) 런타임으로 셀을 순서대로 실행하면 이 교재에서 "기대되는 동작"으로 설명한 실제 출력을 얻을 수 있습니다. 실행 결과를 기록하고 싶다면 이어서 요청해 주십시오.
