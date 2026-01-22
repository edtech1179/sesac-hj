# 필요한 패키지 설치
# pip install langchain langchain-openai

import os
from dotenv import load_dotenv

# [환경 변수 로드]
# API 키라는 비밀 열쇠를 금고(.env)에서 꺼내옵니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. AI 모델(요리사) 준비하기
from langchain_classic.chat_models import ChatOpenAI

# 똑똑한 요리사(gpt-4o-mini)를 고용합니다.
# temperature=0: "창의력은 잠시 넣어두고, 정확하게만 대답해줘"라고 지시합니다.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# 2. 기억력을 가진 대화 체인(ConversationChain) 만들기

# 보통의 AI(LLM)는 금붕어와 같아서 방금 한 이야기도 금방 잊어버립니다.
# 하지만 'ConversationChain'은 대화 내용을 '메모'해두기 때문에 이전 이야기를 기억할 수 있습니다.

from langchain_classic import ConversationChain

# conversation: 기억력을 가진 특별한 관리자입니다.
# verbose=True: 관리자가 속으로 무슨 생각을 하는지(로그) 전부 보여달라는 뜻입니다. (메모하는 과정을 훔쳐볼 수 있어요!)
conversation = ConversationChain(llm=llm, verbose=True)

# 3. 대화 시작해보기

# (1) 첫 번째 정보 주기: 진희네 강아지
print("\n--- 첫 번째 대화: 진희네 강아지 ---")
# 관리자에게 말을 겁니다. "진희는 강아지를 한 마리 키워."
conversation.predict(input="진희는 강아지를 한마리 키우고 있습니다.")

# (2) 두 번째 정보 주기: 영수네 고양이
print("\n--- 두 번째 대화: 영수네 고양이 ---")
# 관리자에게 또 다른 정보를 줍니다. "영수는 고양이를 두 마리 키워."
# 이때 관리자는 첫 번째 정보(진희네 강아지)도 여전히 기억하고 있습니다.
conversation.predict(input="영수는 고양이를 두마리 키우고 있습니다.")

# (3) 기억력 테스트: 총 몇 마리일까?
print("\n--- 세 번째 대화: 기억력 테스트 ---")
# 이제 두 정보를 종합해야 풀 수 있는 문제를 냅니다.
# 만약 기억력이 없다면 "누구요? 무슨 동물이요?"라고 하겠지만,
# 우리의 똑똑한 관리자는 앞의 내용을 다 기억하고 대답해줍니다.
result = conversation.predict(input="진희와 영수가 키우는 동물은 총 몇마리?")

print(f"\n최종 답변: {result}")
