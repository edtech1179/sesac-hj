import os
import json
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드 (API 정보 등 보안이 필요한 설정을 가져옵니다)
load_dotenv()
# 환경변수에서 OPENAI_API_KEY를 가져와 변수에 저장합니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------------
# 1. 라이브러리 설치 및 설정
# ------------------------------------------------------------------------------
# 필요한 라이브러리를 설치하는 명령어입니다. (터미널에서 실행 필요)
# pip install langchain openai langchain-openai langchain-experimental langchainhub ddgs

import langchain
# 랭체인의 디버그 모드를 활성화하여 실행 과정을 상세하게 출력합니다 (verbose보다 더 자세함)
langchain.debug = True
# verbose 모드는 끕니다 (debug 모드가 켜져 있으므로)
langchain.verbose = False

# langchain_classic 모듈에서 필요한 기능들을 임포트합니다.
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent, load_tools, create_openai_functions_agent, create_openai_tools_agent
from langchain_classic.tools import Tool
from langchain_classic.prompts import PromptTemplate
from langchain_classic import LLMChain
from langchain_classic.chains import create_extraction_chain
from langchain_classic.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
# 2. ReAct Agent 예제 (Terminal 도구 사용)
# ------------------------------------------------------------------------------
print("--- ReAct Agent (Terminal) ---")

# LLM 객체 생성 (GPT-4.1-mini 모델 사용, 창의성 0)
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 사용할 도구 로드 (Terminal)
# allow_dangerous_tools=True: 터미널 명령어 실행 허용
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# ReAct 프롬프트 다운로드
prompt = hub.pull("hwchase17/react")

# ReAct 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 에이전트 실행기 생성 (verbose=True로 과정 출력)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행: sample_data 디렉터리 파일 목록 확인 요청 (인코딩 처리 포함)
result = agent_executor.invoke({"input": "sample_data 디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰줘."})
print(result["output"])


# ------------------------------------------------------------------------------
# 3. Custom Tool 만들기 (The_Answer): AI에게 '컨닝 페이퍼'나 '비법 책'을 하나 쥐어주는 것과 같습니다.
# ------------------------------------------------------------------------------
print("\n--- Custom Tool (The_Answer) ---")

# 사용자 정의 함수: 무조건 "42"를 반환하는 함수입니다.
# AI가 스스로 답을 계산하는 것이 아니라, 우리가 정해놓은 답을 그대로 가져오게 합니다.
def my_super_func(param):
    return "42"

# 사용자 정의 도구 생성
tools = [
    Tool.from_function(
        func=my_super_func,  # AI가 이 도구를 사용할 때 실행할 실제 파이썬 함수 (위에서 만든 함수)
        name="The_Answer",   # AI가 식별하는 도구의 이름 ('정답' 도구야)
        description="생명, 우주, 그리고 모든 것에 대한 궁극적인 질문의 답" # AI가 '언제' 이 도구를 써야 할지 알려주는 설명서
    ),
]

# 에이전트 생성 및 실행
# 위에서 만든 'The_Answer' 도구를 쥐어줍니다.
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 질문: "이 세계의 진리를 알려주세요"
# AI는 자신의 지식으로 답하기 애매한 질문을 받으면, 도구 설명을 훑어봅니다.
# "궁극적인 질문의 답"이라는 설명을 보고 "아, The_Answer 도구를 쓰면 되겠구나!"라고 판단하여 실행합니다.
result = agent_executor.invoke({"input": "이 세계의 진리를 알려주세요"})
print(result)


# ------------------------------------------------------------------------------
# 4. LLMChain을 Tool로 사용하기 (Summarizer): AI에게 '전문가 비서'를 붙여주는 것과 같습니다.
# ------------------------------------------------------------------------------
print("\n--- LLMChain as a Tool (Summarizer) ---")

# 요약 템플릿 정의: 비서에게 내릴 업무 매뉴얼입니다.
summarize_template = """아래의 글을 결론만 한 마디로 요약해 주세요.
{input}
"""
# 프롬프트 템플릿 생성: 매뉴얼을 깔끔한 서식으로 만듭니다.
summarize_prompt = PromptTemplate(
    input_variables=["input"], # 입력받을 변수 이름
    template=summarize_template, # 사용할 템플릿
)

# 요약 작업을 수행할 LLMChain 생성: '요약 전문 비서'를 채용합니다.
# 이 체인은 오직 '요약'만 할 줄 아는 전문 AI입니다.
chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

# LLMChain을 도구로 래핑: 채용한 비서를 'Summarizer'라는 이름으로 메인 AI(팀장)에게 소개합니다.
tools = [ 
    Tool.from_function(
        func=summarize_chain.run, # 팀장이 이 도구를 쓰면 실제로 '요약 비서'가 일을 합니다.
        name="Summarizer",        # 도구 이름 (팀장이 부를 이름)
        description="Text summarizer" # 도구 설명 (팀장이 언제 이 비서를 부를지 판단하는 기준)
    )
]

# 에이전트 생성 및 실행: 메인 AI(팀장)를 생성하고 요약 비서를 배정합니다.
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 요약할 텍스트: 팀장이 처리해야 할 원문 데이터
text = """다음을 요약해 주세요.
안녕하세요! 저는 ChatGPT라고 불리는 AI 언어 모델입니다.
OpenAI가 개발한 GPT-3.5 아키텍처를 기반으로 합니다.
저는 자연어 이해와 생성을 전문으로 하며, 다양한 주제에 대한 질문에 답하거나, 대화를 나누는 것을 잘합니다.
제 트레이닝 데이터는 2021년 9월까지의 정보를 기반으로 하기 때문에, 그 이후의 사건에 대해서는 지식이 없습니다.
하지만, 가능한 한 도움을 드리기 위해 노력할 것입니다.
질문이나 대화, 정보 공유 등, 어떤 도움이든 편하게 말씀해 주세요! 잘 부탁드립니다."""

# 에이전트 실행: "이 글 좀 요약해줘"라고 시킵니다.
# 메인 AI는 직접 요약하지 않고, "Summarizer 도구(비서)"에게 일을 토스합니다.
result = agent_executor.invoke({"input": text})
print(result)


# ------------------------------------------------------------------------------
# 5. OpenAI Functions Agent (Terminal): OpenAI의 특수 기능인 'Function Calling'을 사용하는 똑똑한 에이전트
# ------------------------------------------------------------------------------
print("\n--- OpenAI Functions Agent (Terminal) ---")

# LLM 생성 (Functions 기능을 지원하는 모델 사용)
# OpenAI 모델들은 "함수 호출"을 아주 잘하도록 특별히 훈련되어 있습니다.
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 도구 로드 (Terminal)
# 여기서도 터미널 도구를 사용합니다.
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# OpenAI Functions Agent용 프롬프트 다운로드
# 일반적인 ReAct 프롬프트와 달리, OpenAI 모델에 최적화된 프롬프트를 가져옵니다.
prompt = hub.pull("hwchase17/openai-functions-agent")

# OpenAI Functions Agent 생성
# create_react_agent 대신 create_openai_functions_agent를 사용합니다.
# 이 에이전트는 도구를 언제, 어떻게 써야 할지 더 정확하게 판단합니다.
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행: 파일 목록 확인 요청
# Functions Agent는 함수 호출 방식이 더 정교하지만, 터미널 명령어 형식에 따라 에러가 발생할 수 있음 (노트북 결과 참고)
# agent_executor.invoke({"input": "sample_data디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰서 해줘."})


# ------------------------------------------------------------------------------
# 6. DuckDuckGo Search (ddgs) 활용하기
# ------------------------------------------------------------------------------
print("\n--- OpenAI Functions Agent (DuckDuckGo Search) ---")

# DuckDuckGo 검색 도구 로드
# 터미널 대신 'ddg-search' 도구를 로드하여 인터넷 검색 기능을 부여합니다.
# 이 도구를 사용하면 AI가 실시간 정보를 검색할 수 있게 됩니다.
tools = load_tools(["ddg-search"])

# OpenAI Functions Agent 생성
# 검색 도구를 사용하는 똑똑한 에이전트를 만듭니다.
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행: 서울과 부산 날씨 검색 요청
# AI는 자신의 지식에 없는 '현재 날씨' 정보를 얻기 위해 'ddg-search' 도구를 자동으로 호출합니다.
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})
print(result["output"])


# ------------------------------------------------------------------------------
# 7. OpenAI Tools Agent: 더 일반적이고 강력한 최신 에이전트 방식
# ------------------------------------------------------------------------------
print("\n--- OpenAI Tools Agent ---")

# LLM 생성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 도구 로드 (DuckDuckGo Search)
tools = load_tools(["ddg-search"])

# OpenAI Tools Agent용 프롬프트 다운로드
# Functions Agent와 비슷하지만, 'Tools'라는 더 포괄적인 개념을 사용하는 최신 프롬프트입니다.
prompt = hub.pull("hwchase17/openai-tools-agent")

# OpenAI Tools Agent 생성
# create_openai_tools_agent를 사용하여 최신 방식의 에이전트를 생성합니다.
# parallel function calling (병렬 함수 호출) 등 더 발전된 기능을 지원합니다.
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행: 날씨 정보 검색
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})
print(result["output"])


# ------------------------------------------------------------------------------
# 8. Extraction (Structured Output): 텍스트에서 원하는 정보만 쏙쏙 뽑아내기
# ------------------------------------------------------------------------------
print("\n--- Extraction Chain ---")

# 추출할 데이터의 스키마 정의 (JSON Schema 형식)
# AI에게 "이런 모양으로 데이터를 정리해줘"라고 틀을 잡아주는 단계입니다.
schema = {
    "properties": {
        "person_name": {"type": "string"},      # 사람이름 (문자열)
        "person_age": {"type": "integer"},      # 나이 (정수)
        "person_hair_color": {"type": "string"}, # 머리색 (문자열)
        "dog_name": {"type": "string"},         # 개 이름 (문자열)
        "dog_breed": {"type": "string"},        # 개 품종 (문자열)
    },
    "required": ["person_name"], # 필수값 지정 (이 정보는 꼭 찾아내라)
}

# 텍스트 데이터: 정보를 추출할 원문
text = """Alex is 5 feet tall, Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek."""

# Extraction Chain 생성
# create_extraction_chain: 텍스트와 스키마를 주면, 스키마에 맞는 데이터만 추출하는 전문 체인을 만듭니다.
chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
chain = create_extraction_chain(schema, chat)

# 체인 실행 및 결과 출력
# invoke(text)를 호출하면 텍스트를 분석하여 스키마에 정의된 키-값 쌍으로 리스트를 반환합니다.
people = chain.invoke(text)
print(json.dumps(people, indent=2)) # 보기 좋게 들여쓰기하여 출력


# ------------------------------------------------------------------------------
# 9. Evaluation (평가): AI의 대답이 얼마나 정확한지 채점하기
# ------------------------------------------------------------------------------
print("\n--- Evaluation ---")

# 평가용 LLM 생성: 채점관 역할을 할 AI
chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 'qa' (Question Answering) 평가기 로드
# QA 평가기는 "질문 + 정답(기준) + 예측값(AI의 답)"을 비교하여 점수를 매깁니다.
evaluator = load_evaluator("qa", eval_11m=chat)

# evaluate_strings: 평가 실행 함수
result = evaluator.evaluate_strings(
    input="""나는 시장에 가서 사과 10개를 샀어.
사과 2개를 이웃에게 주고, 2개를 수리공에게 주었어.
그리고 사과 5개를 더 사서 1개는 내가 먹었어.
나는 몇 개의 사과를 가지고 있었니?""", # 질문 (문제)
    prediction="""먼저 사과 10개로 시작했어.
이웃에게 2개, 수리공에게 2개를 나누어 주었으므로 사과가 6개가 남았어.
그런 다음 사과 5개를 더 사서 이제 사과가 11개가 되었어.
마지막으로 사과 1개를 먹었으므로 사과 10개가 남게 돼.""", # AI가 내놓은 답 (채점 대상)
    reference="10개", # 실제 정답 (채점 기준)
)

# 평가 결과 출력
# CORRECT: 정답 여부, SCORE: 점수, REASONING: 채점 이유 등이 포함됩니다.
print(result)
