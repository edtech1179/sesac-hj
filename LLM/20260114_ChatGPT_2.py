# ==========================================
# [개요] ChatGPT API 활용 예제 스크립트
#
# 이 스크립트는 OpenAI의 Chat Completions API를 사용하는 다양한 방법을 다룹니다.
# 
# 주요 흐름:
# 1. 토큰(Token) 계산: tiktoken 라이브러리를 사용하여 텍스트가 몇 개의 토큰으로 변환되는지 확인합니다.
# 2. 환경 설정: .env 파일에서 API 키를 불러옵니다.
# 3. 기본 채팅: 간단한 질문과 답변을 주고받습니다.
# 4. 멀티턴 대화: 이전 대화 내용을 포함하여 문맥을 유지하는 대화를 합니다.
# 5. 스트리밍: 응답을 한꺼번에 받지 않고 실시간으로 조금씩 받아 출력합니다.
# 6. 함수 호출(Function Calling): AI가 외부 함수를 실행해야 할 때를 판단하고, 결과를 받아 최종 답변을 생성하는 과정을 실습합니다.
# ==========================================

# --------------------------------------------------------------------------------
# 1. 토큰(Token) 계산하기 (tiktoken 라이브러리)
# --------------------------------------------------------------------------------
# LLM은 텍스트를 글자 단위가 아닌 '토큰' 단위로 처리합니다.
# 입력과 출력의 길이는 이 토큰 수에 제한을 받으며, 요금도 토큰 단위로 부과됩니다.

# 필요 라이브러리 설치 (최초 1회)
# pip install tiktoken

import tiktoken 

# 영어 문장 예시
text = "It's easy to make something cool with LLMs, but very hard to make something production-ready wiht them."

# gpt-4.1-mini 모델이 사용하는 인코딩 방식(토크나이저)을 가져옵니다.
encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

# 텍스트를 토큰 ID의 리스트로 변환합니다.
tokens = encoding.encode(text)

# 변환된 토큰의 개수를 출력합니다.
print(f"영어 문장 토큰 수: {len(tokens)}")

# 한글 문장 예시 (한글은 영어보다 토큰을 더 많이 사용하는 경향이 있습니다)
text ="LLM을 사용해서 멋져 보이는 것을 만들기는 쉽지만, 프로덕션 수준으로 만들어 내기는 매우 어렵다."

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
tokens = encoding.encode(text)
print(f"한글 문장 토큰 수: {len(tokens)}")


# --------------------------------------------------------------------------------
# 2. Chat Completions API 사용 준비
# --------------------------------------------------------------------------------
# OpenAI API 키를 환경 변수에서 안전하게 로드합니다.

import os
from dotenv import load_dotenv

# .env 파일에 저장된 환경 변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API 키를 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --------------------------------------------------------------------------------
# 3. Chat Completions API 기본 사용법
# --------------------------------------------------------------------------------
# OpenAI 클라이언트를 생성하고 가장 기본적인 대화를 요청합니다.

# pip install openai

from openai import OpenAI

# 클라이언트 인스턴스 생성 (API 키는 환경변수에서 자동으로 기져옵니다)
client = OpenAI()

# API 호출: 모델에게 메시지를 보내고 응답을 받습니다.
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        # system: AI의 역할(페르소나)을 정의합니다.
        {"role": "system", "content": "You are a helpful assistant."},
        # user: 사용자가 입력하는 질문입니다.
        {"role": "user", "content": "Hello! I'm John."}
    ]
)

# 응답 객체 전체 출력 (구조 확인용)
print("\n--- 기본 응답 객체 ---")
print(response)

# 응답 객체를 보기 좋은 JSON 형태로 출력
print("\n--- JSON 포맷 응답 ---")
print(response.model_dump_json(indent=2))


# --------------------------------------------------------------------------------
# 4. 대화 이력(Context)을 활용한 멀티턴 대화
# --------------------------------------------------------------------------------
# API는 기본적으로 '상태가 없음(Stateless)'입니다. 즉, 이전 대화를 기억하지 못합니다.
# 대화의 맥락을 유지하려면 이전 질문과 답변을 매번 함께 보내줘야 합니다.

print("\n--- 멀티턴 대화 (이름 기억하기) ---")
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! I'm John."}, # 사용자가 이름을 말함
        {"role": "assistant", "content": "Hello John! How can I assist you today?"}, # AI가 대답했다고 가정하고 기록 추가
        {"role": "user", "content": "Do you know my name?"} # 다시 질문
    ]
)

# AI가 문맥을 파악하고 John이라고 대답하는지 확인
print(response.model_dump_json(indent=2))


# --------------------------------------------------------------------------------
# 5. 스트리밍(Streaming) 방식으로 응답 받기
# --------------------------------------------------------------------------------
# 긴 응답을 한 번에 기다리지 않고, 타자기 치듯 글자가 생성되는 대로 실시간으로 받습니다.
# stream=True 옵션을 사용합니다.

print("\n--- 스트리밍 응답 ---")
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! I'm John."},
    ],
    stream=True, # 스트리밍 활성화
)

# 스트림 객체를 순회하며 조각(chunk) 단위로 출력합니다.
for chunk in response:
    choice = chunk.choices[0]
    # finish_reason이 None인 동안(생성 중일 때만 아래를 실행) 계속 내용이 들어옵니다.
    if choice.finish_reason is None: 
        # end=""를 사용하여 줄바꿈 없이 이어서 출력
        print(choice.delta.content, end="")
print() # 마지막 줄바꿈


# --------------------------------------------------------------------------------
# 6. 함수 호출 (Function Calling)
# --------------------------------------------------------------------------------
# LLM이 직접 할 수 없는 일(예: 실시간 날씨 조회, DB 쿼리 등)을 
# 외부 함수를 통해 수행할 수 있도록 연결해주는 기능입니다.
#
# [흐름]
# 1. 사용할 함수의 정의(이름, 파라미터 등)를 AI에게 알려줍니다.
# 2. 사용자가 질문을 던집니다 (예: "서울 날씨 어때?").
# 3. AI는 자신이 답할 수 없음을 알고, '함수 호출'이 필요하다는 응답을 보냅니다.
# 4. 코드는 AI가 요청한 함수(get_current_weather)를 실제로 실행하여 결과를 얻습니다.
# 5. 함수의 실행 결과를 AI에게 메시지로 추가하여 다시 보냅니다.
# 6. AI는 함수 결과를 바탕으로 사용자에게 최종 자연어 답변을 합니다.

import json

# (1) 실제 실행될 함수 정의 (가짜 데이터 반환)
def get_current_weather(location, unit="celsius"):
    """특정 위치의 현재 날씨를 가져오는 함수 (예제용 더미 데이터)"""
    weather_info = {
        "location": location,
        "temperature": "25",
        "unit": "celsius",
        "forecast": ["sunny", "windy"],
    }
    # 결과를 JSON 문자열로 반환
    return json.dumps(weather_info)

# (2) AI에게 알려줄 함수 명세 (스키마) 정의
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The location to get the weather for"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"]
        }
    }
]

# (3) 사용자 질문
messages = [{"role": "user", "content": "What's the weather like in Seoul?"}]

# (4) 1차 호출: 함수 정의(functions)를 포함해서 요청
print("\n--- Function Calling: 1차 호출 (함수 판단) ---")
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages,
    functions=functions # 사용 가능한 함수 목록 전달
)

# 응답 확인: AI가 텍스트 대신 'function_call'을 요청했는지 확인
print(response.model_dump_json(indent=2))

response_message = response.choices[0].message

# (5) AI가 함수 호출을 요청했을 경우, 실제로 함수 실행
# 사용 가능한 함수 매핑
available_functions = {
    "get_current_weather": get_current_weather,
}

if response_message.function_call:
    # 호출할 함수 이름 확인
    function_name = response_message.function_call.name
    # 실제 함수 객체 가져오기
    function_to_call = available_functions[function_name]
    # AI가 추출한 파라미터(인자) 가져오기
    function_args = json.loads(response_message.function_call.arguments)
    
    # 함수 실행
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )
    
    print(f"\n--- 함수 실행 결과: {function_response} ---")

    # (6) 대화 내역에 AI의 '함수 호출 요청' 메시지 추가
    messages.append(response_message)

    # (7) 대화 내역에 '함수 실행 결과' 메시지 추가 (role: function)
    messages.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        }
    )

    print("\n--- 업데이트된 메시지 목록 ---")
    print(messages)

    # (8) 2차 호출: 함수 결과를 포함한 대화 내역을 다시 AI에게 전달
    print("\n--- Function Calling: 2차 호출 (최종 답변) ---")
    second_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    # (9) 최종 결과 출력 (자연어로 된 날씨 설명)
    print(second_response.model_dump_json(indent=2))
