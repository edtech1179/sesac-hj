import warnings
# 경고 메시지 무시 설정 (실행 시 불필요한 경고가 많이 뜨는 것을 방지)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 필요한 패키지 설치 명령어 (참고용)
# pip install langchain openai tiktoken pypdf faiss-cpu sentence-transformers tf-keras
# ------------------------------------------------------------------------------

import os
from dotenv import load_dotenv

# .env 파일 로드: API 키 등 보안이 필요한 환경 변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader

# ------------------------------------------------------------------------------
# 1. 문서 로드 (Document Loading)
# ------------------------------------------------------------------------------
# 분석할 PDF 파일의 경로를 지정합니다. (톰 소여의 모험)
pdf_path = "C:\\강의 교안\\LLM\\The_Adventures_of_Tom_Sawyer.pdf"

# PDF 로더 인스턴스를 생성하고 파일을 로드하여 텍스트 문서 객체로 변환합니다.
loader = PyPDFLoader(pdf_path)
document = loader.load()

# 문서 내용 확인용 코드 (노트북에서 출력 확인용, 필요 시 주석 해제)
# print(document[5].page_content[:5000])

# ------------------------------------------------------------------------------
# 2. 벡터 저장소 및 임베딩 (Vector Store & Embeddings)
# ------------------------------------------------------------------------------
# FAISS: 고성능 벡터 유사도 검색 라이브러리
# OpenAIEmbeddings: 텍스트를 벡터(숫자)로 변환해주는 OpenAI 모델
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import OpenAIEmbeddings

# OpenAI 임베딩 모델을 초기화합니다.
embeddings = OpenAIEmbeddings()

# 로드한 PDF 문서(document)를 임베딩 벡터로 변환하여 FAISS DB에 저장합니다.
# 이로써 톰 소여의 모험 책 내용이 검색 가능한 상태가 됩니다.
db = FAISS.from_documents(document, embeddings)

# --- 임베딩 변환 예제 ---
# (아래 text는 단순히 임베딩이 어떻게 되는지 보여주는 예시이며, db에는 저장되지 않습니다)
text = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"

# 텍스트를 쿼리 임베딩(숫자 리스트)으로 변환
text_embedding = embeddings.embed_query(text)
print(f"OpenAI Embedding Sample (앞 50개): {text_embedding[:50]}")

# ------------------------------------------------------------------------------
# 3. 다른 임베딩 모델 사용 예제 (HuggingFace)
# ------------------------------------------------------------------------------
from langchain_classic.embeddings import HuggingFaceEmbeddings

# HuggingFace의 오픈소스 모델(all-MiniLM-L6-v2)을 사용하여 임베딩 모델 교체
# 로컬에서 실행되므로 API 비용이 들지 않는 장점이 있습니다.
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
text_embedding_hf = hf_embeddings.embed_query(text)
print(f"HuggingFace Embedding Sample (앞 50개): {text_embedding_hf[:50]}")

# ------------------------------------------------------------------------------
# 4. 질의응답 체인 구성 (RetrievalQA)
# ------------------------------------------------------------------------------
from langchain_classic.chat_models import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# 답변을 생성할 LLM(언어 모델) 초기화
# model_name: 사용할 모델 지정 (gpt-4.1-mini는 예시이며, 실제로는 gpt-4o-mini 등을 사용 권장)
# temperature: 0으로 설정하여 무작위성을 줄이고 사실적인 답변 유도
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 앞서 만든 벡터 저장소(db)를 검색기(retriever)로 변환합니다.
# 질문이 들어오면 이 retriever가 유사한 문서를 찾아줍니다.
retriever = db.as_retriever()

# RetrievalQA 체인 생성 (검색 + 답변 생성)
# chain_type="stuff": 검색된 문서들을 프롬프트에 전부 채워넣어(Stuff) LLM에 전달하는 방식
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# ------------------------------------------------------------------------------
# 5. 질문 및 결과
# ------------------------------------------------------------------------------
# 질문: 진희가 키우고 있는 동물은 무엇인가?
#
# [중요] 결과가 "알 수 없음"으로 나오는 이유:
# 현재 'db'에는 '톰 소여의 모험(PDF)' 내용만 저장되어 있습니다.
# 위 코드에서 변수 'text'로 진희 이야기를 작성했지만, 이 내용은 'db'에 add_documents()로 추가되지 않았습니다.
# 따라서 Retriever는 진희에 대한 정보를 찾을 수 없습니다.
query = "진희가 키우고 있는 동물은 무엇인가?"
result = qa({"query": query})

print(f"질문: {query}")
print(f"답변: {result['result']}")
