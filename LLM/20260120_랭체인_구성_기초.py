import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma  # ChromaDB import
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
# 1. 환경 변수 설정 (Environment Setup)
# ------------------------------------------------------------------------------

# .env 파일에서 환경 변수를 로드합니다.
# API 키와 같은 민감한 정보는 소스 코드에 직접 작성하지 않고 .env 파일로 관리하는 것이 보안상 좋습니다.
load_dotenv()

# 환경 변수에서 OpenAI API 키를 가져옵니다.
# 이 키는 나중에 OpenAIEmbeddings 등을 사용할 때 자동으로 활용됩니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------------
# 2. 필수 라이브러리 설치 안내 (Dependencies)
# ------------------------------------------------------------------------------
# 아래 라이브러리들이 설치되어 있어야 이 코드가 정상적으로 실행됩니다.
# 터미널에서 아래 명령어를 실행하여 설치할 수 있습니다.
# pip install langchain openai langchain-openai langchain-community langchain_classic pydantic Gitpython tiktoken chromadb

# ------------------------------------------------------------------------------
# 3. 데이터 로드 (Data Loading)
# ------------------------------------------------------------------------------
# GitLoader를 사용하여 GitHub 저장소의 파일들을 불러옵니다.

# 파일 필터링 함수 정의
def file_filter(file_path):
    """
    파일 경로를 입력받아 확장자가 .md(마크다운)인 파일만 True를 반환합니다.
    즉, 마크다운 파일만 골라서 로드하기 위한 필터입니다.
    """
    return file_path.endswith(".md")

print("1. Git 저장소에서 문서를 로드 중입니다...")

# GitLoader 초기화
# clone_url: 복제할 GitHub 저장소 주소
# repo_path: 로컬에 저장소가 복제될 경로 (여기서는 현재 디렉토리의 langchain 폴더)
# file_filter: 앞서 정의한 필터 함수를 적용하여 원하는 파일만 로드
# branch: 가져올 브랜치 이름 (여기서는 'master' 브랜치 사용)
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    file_filter=file_filter,
    branch="master", 
)

# 문서 로드 실행
# 지정한 저장소에서 조건에 맞는 파일들을 모두 읽어와 raw_docs 변수에 저장합니다.
try:
    raw_docs = loader.load()
    print(f"   -> 총 {len(raw_docs)}개의 문서를 성공적으로 로드했습니다.")
except Exception as e:
    print(f"   -> [오류] 문서 로드 실패: {e}")
    print("      './langchain' 폴더를 삭제하고 다시 시도해 보시거나 git 설치 여부를 확인해 보세요.")

# ------------------------------------------------------------------------------
# 4. 문서 분할 (Document Splitting)
# ------------------------------------------------------------------------------
# 가져온 문서들이 너무 길 수 있으므로, 처리하기 좋은 크기로 자릅니다.

print("\n2. 문서를 청크(chunk) 단위로 분할 중입니다...")

# CharacterTextSplitter 초기화
# chunk_size: 문서를 자를 최대 크기 (글자 수 기준 1000자)
# chunk_overlap: 잘린 문서들 간의 중복 허용 구간 (여기서는 0으로 설정하여 중복 없음)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# 문서 분할 실행
# 로드한 원본 문서(raw_docs)를 설정한 크기에 맞춰 분할하여 docs 변수에 저장합니다.
docs = text_splitter.split_documents(raw_docs)
print(f"   -> 분할 완료: 총 {len(docs)}개의 청크가 생성되었습니다.")


# ------------------------------------------------------------------------------
# 5. 텍스트 임베딩 (Text Embedding)
# ------------------------------------------------------------------------------
# 텍스트를 AI가 이해할 수 있는 숫자 벡터(Vector)로 변환하는 과정입니다.

print("\n3. 임베딩 모델을 초기화합니다 (OpenAIEmbeddings)...")
# OpenAI의 임베딩 모델 초기화
# 기본적으로 'text-embedding-ada-002' 또는 최신 모델이 사용됩니다.
embeddings = OpenAIEmbeddings()

# 쿼리(질문) 임베딩 테스트
query = "AWS의 S3에서 데이터를 읽기 위한 DocumentLoader가 있나요? "
vector = embeddings.embed_query(query)
print(f"   -> 테스트 쿼리 임베딩 완료. 벡터 차원: {len(vector)}")
# print(f"   -> 벡터 일부: {vector[:100]}") # 너무 길어서 주석 처리

# ------------------------------------------------------------------------------
# 6. 벡터 저장소 생성 (Vector Store - ChromaDB)
# ------------------------------------------------------------------------------
# 분할된 문서(docs)를 임베딩하여 ChromaDB라는 벡터 데이터베이스에 저장합니다.
# 이렇게 하면 나중에 "의미 기반 검색"이 가능해집니다.

print("\n4. 문서들을 벡터로 변환하여 ChromaDB에 저장 중입니다...")
# Chroma.from_documents 함수는 문서 리스트와 임베딩 모델을 받아 자동으로 벡터화하고 DB를 생성합니다.
# 이 과정은 문서 양에 따라 시간이 걸릴 수 있습니다.
db = Chroma.from_documents(docs, embeddings)
print("   -> ChromaDB 저장 완료.")

# ------------------------------------------------------------------------------
# 7. 검색기(Retriever) 설정
# ------------------------------------------------------------------------------
# 저장된 DB를 검색 가능한 'Retriever' 객체로 변환합니다.

print("\n5. 검색기(Retriever)를 설정하고 테스트 검색을 수행합니다...")
retriever = db.as_retriever()

# 테스트 검색 실행
# 앞서 설정한 질문(query)과 관련된 문서를 DB에서 찾아옵니다.
context_docs = retriever.invoke(query)
print(f"   -> 검색된 관련 문서 개수: {len(context_docs)}")

if context_docs:
    first_doc = context_docs[0]
    print(f"   -> 가장 유사한 문서의 메타데이터: {first_doc.metadata}")
    print(f"   -> [문서 내용 일부 미리보기]:\n{first_doc.page_content[:200]}...")


# ------------------------------------------------------------------------------
# 8. 질의응답 체인 (RetrievalQA Chain)
# ------------------------------------------------------------------------------
# 검색된 문서를 바탕으로 LLM(ChatGPT)이 답변을 생성하도록 체인을 구성합니다.

print("\n6. RetrievalQA 체인을 구성하여 최종 답변을 생성합니다...")

# LLM 모델 초기화 (gpt-4o-mini 등 사용)
# temperature=0: 답변의 일관성을 위해 무작위성을 0으로 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# RetrievalQA 체인 생성
# chain_type="stuff": 검색된 문서들을 모두 프롬프트에 '채워 넣어(stuff)' 답변을 생성하는 방식
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 질문에 대한 답변 생성 실행
response = qa_chain.invoke(query)

print("\n" + "="*50)
print(f"질문: {query}")
print("-" * 50)
print(f"답변:\n{response['result']}")
print("="*50)
