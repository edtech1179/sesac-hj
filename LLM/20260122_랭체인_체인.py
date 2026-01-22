# 필요한 패키지 설치
# pip install langchain langchain-openai duckduckgo-search wikipedia

import os
from dotenv import load_dotenv

# [비밀번호 꺼내오기]
# .env라는 금고에서 API 키(열쇠)를 몰래 꺼내옵니다.
# 이렇게 해야 내 비밀번호가 다른 사람에게 노출되지 않아요.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# [필요한 도구들]
# 요리할 때 칼, 도마가 필요하듯 랭체인을 쓰려면 아래 도구들이 필요합니다.
from langchain_classic import LLMChain
from langchain_classic import PromptTemplate
from langchain_classic.chat_models import ChatOpenAI

# 1. 요리사(AI) 부르기
# temperature=0: "창의적인 요리 말고, 레시피 그대로 정확하게 만들어줘"라고 부탁하는 겁니다. (0이면 딴소리를 안 해요)
# model_name="gpt-4o-mini": 손이 빠르고 똑똑한 'gpt-4o-mini' 요리사를 고용했습니다.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")


# 2. 첫 번째 미션: 수도 이름 맞추기 (단순 심부름)

# (1) 빈칸메우기 종이(PromptTemplate) 만들기
# 매번 "미국의 수도는?" "한국의 수도는?" 하고 다 쓰는 건 귀찮죠?
# 그래서 "____의 수도는 어디야?"라고 구멍이 뚫린 종이를 미리 만들어둡니다.
# 나중에 구멍({country})에 "대한민국"만 쏙 넣으면 문장이 완성되니까요!
prompt = PromptTemplate(
    input_variables=["country"],  # 구멍의 이름은 'country'입니다.
    template="{country}의 수도는 어디야?",  # 구멍 뚫린 문장입니다.
)

# (2) 작업 반장님(LLMChain) 지정하기
# 요리사(llm)에게 "이 종이(prompt)에 적힌 대로 대답해줘"라고 시키는 '반장님'입니다.
# 반장님이 요리사와 주문서를 딱 연결해줍니다.
chain = LLMChain(llm=llm, prompt=prompt)

# (3) 반장님께 시키기
# "여기에 '대한민국' 써서 요리사한테 물어봐주세요!"라고 하는 겁니다.
print("--- 첫 번째 결과 ---")
print(chain.run("대한민국"))


# 3. 두 번째 미션: 이어달리기 (SequentialChain)
# 이번엔 혼자서 안 끝납니다. "번역"을 하고 -> 그 결과로 "요약"을 하는 이어달리기입니다.

# (1) 첫 번째 주자: 번역 담당
# 영어 문장을 주면 한글로 바꿔주는 역할을 맡았습니다.
prompt1 = PromptTemplate(
    input_variables=['sentence'],  # "영어 문장"을 받습니다.
    template="다음 문장을 한글로 번역하세요.\n\n{sentence}"
)

# 첫 번째 주자(chain1)입니다.
# output_key="translation": 자기가 뛴 결과물(번역본)에 "translation"이라는 이름표를 붙여서 다음 사람에게 넘깁니다.
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="translation")


# (2) 두 번째 주자: 요약 담당
# 앞사람이 넘겨준 "번역본"을 받아서 짧게 요약하는 역할을 맡았습니다.
prompt2 = PromptTemplate.from_template(
    "다음 문장을 한 문장으로 요약하세요.\n\n{translation}" # {translation}은 앞사람이 준 이름표랑 똑같아야 받을 수 있어요!
)

# 두 번째 주자(chain2)입니다.
# output_key="summary": 이 친구의 최종 결과물에는 "summary"라는 이름표를 붙입니다.
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")


# (3) 이어달리기 팀(SequentialChain) 결성
from langchain_classic.chains import SequentialChain

# 전체 팀(all_chain)을 만듭니다.
all_chain = SequentialChain(
    chains=[chain1, chain2],  # 팀원 배치: 1번 주자(번역) -> 2번 주자(요약)
    input_variables=["sentence"],  # 시합 시작할 때 필요한 것: 영어 문장
    output_variables=["translation", "summary"],  # 경기 끝나고 받고 싶은 것들: 번역본과 요약본
)

# 시합에 쓸 바통 (영어 문장)
sentence = """
One limitation of LLMs is their lack of contextual information
(e.g., access to some specific documents or emails).
You can combat this by giving LLMs access to the specific external data.
For this, you first need to load the external data with a document loader.
LangChain provides a variety of loaders for different types of documents ranging
from PDFs and emails to websites and YouTube videos.
"""

# 경기 시작! (All Chain 실행)
print("\n--- 이어달리기 결과 ---")
# 팀장님(all_chain)에게 바통(sentence)을 넘겨주면 알아서 선수들이 뜁니다.
result = all_chain(sentence)

# 결과 발표
print(f"원래 문장:\n{result['sentence']}\n") # 처음에 준 것
print(f"1번 주자(번역):\n{result['translation']}\n") # 첫 번째 결과
print(f"2번 주자(요약):\n{result['summary']}") # 두 번째 결과
