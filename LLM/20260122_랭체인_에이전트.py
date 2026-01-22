# 필요한 패키지 설치
# pip install langchain langchain-openai wikipedia numexpr

import os
from dotenv import load_dotenv

# [비밀번호 꺼내오기]
# 에이전트도 일을 하려면 돈(API Key)이 필요하겠죠? 금고(.env)에서 꺼내옵니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. AI 두뇌(Brain) 준비하기
from langchain_classic.chat_models import ChatOpenAI

# 똑똑한 팀장님(gpt-4o-mini)을 모셔옵니다.
# temperature=0: "상상하지 말고 팩트 체크만 정확히 해주세요"라고 요청합니다.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")


# 2. 에이전트(해결사)에게 쥐어줄 도구(Tools) 준비하기
from langchain_classic.agents import load_tools

# 에이전트는 혼자서 모든 걸 알 수 없습니다. 그래서 '도구'가 담긴 배낭을 챙겨줘야 합니다.
# wikipedia: 세상의 지식이 담긴 "백과사전"
# llm-math: 복잡한 나이 계산을 대신해줄 "계산기"
tools = load_tools(["wikipedia", "llm-math"], llm=llm)


# 3. 에이전트(해결사) 임명하기
from langchain_classic.agents import initialize_agent
from langchain_classic.agents import AgentType

# 이제 도구(tools)와 두뇌(llm)를 합쳐서 "만능 해결사(Agent)"를 만듭니다.
# agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION:
# "설명서만 보고(Description) 그때그때 상황에 맞춰(React) 도구를 골라 쓰는 방식"입니다.
# 즉, 미리 학습하지 않아도(Zero-shot) "이건 계산기가 필요하겠군", "이건 위키 검색이 필요하겠군" 하고 스스로 판단합니다.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    description="계산이 필요할 때 사용", # 에이전트가 어떤 상황에서 활약할지 알려줍니다.
    verbose=True # 해결사가 일하는 과정(생각하는 소리)을 다 들려줍니다.
)

# 4. 미션 주기
# 미션: "트럼프 나이 계산해줘"
# (1) 트럼프가 언제 태어났는지 검색해야 하고 (Wikipedia)
# (2) 지금 연도(2026)에서 태어난 연도를 빼야 합니다 (Calculator)
# 에이전트가 이 순서를 스스로 생각해서 실행할 겁니다.
print("\n--- 에이전트 미션 시작 ---")
agent.run("트럼프가 태어난 해는? 2026년도 현재 트럼프는 몇 살?")
