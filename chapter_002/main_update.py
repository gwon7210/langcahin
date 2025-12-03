# 필요한 라이브러리 호출
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# .env 파일에 저장된 API KEY 등을 자동으로 불러오는 부분
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please set environment variables manually.", ImportWarning)

def main():
    # 웹페이지 기본 설정
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")

    #채팅 이력 초기화: message_history가 없다면 새로 생성
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 2️. 사용할 LLM 모델 설정
    #   (기본 설정으로는 gpt-4o-mini 호출된다.)
    llm = ChatOpenAI(
        temperature=0          
    )
 
    # 3️. LLM에 전달할 메시지 템플릿 정의
    # - system: 어시스턴트의 성격 정의 (기본 규칙)
    # - history: 지금까지 대화 기록이 자동으로 들어갈 자리
    # - user: 사용자가 지금 입력한 최신 메시지
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_input}")
    ])

    # LLM 응답을 텍스트로 변환해주는 파서
    output_parser = StrOutputParser()

    # 4️. 사용자의 질문을 ChatGPT에 전달하고, 응답을 받아오는 연속적인 처리(체인)를 생성
    #    각 요소를 | (파이프)로 연결해서 연속적인 처리를 만드는 것이 LCEL의 특징
    chain = prompt | llm | output_parser

    # 5️. Streamlit 전용 입력 창
    if user_input := st.chat_input("궁금한 것을 입력해줘!"):
        # 입력을 받으면 이 부분이 실행됩니다.
        with st.spinner("ChatGPT가 답변 중 ..."):
            # invoke 실행 시 히스토리와 사용자 입력을 프롬프트에 전달
            response = chain.invoke({
                "history": st.session_state.message_history,  # placeholder에 들어감
                "user_input": user_input
            })

        # 사용자의 질문을 이력에 추가 ('user'는 사용자의 질문을 의미한다).
        st.session_state.message_history.append(("user", user_input))
        
        # ChatGPT의 답변을 이력에 추가 ('assistant'는 ChatGPT의 답변을 의미한다)
        st.session_state.message_history.append(("assistant", response))

    # UI에 과거 대화 모두 출력
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

if __name__ == '__main__':
    main()

 
 

 