# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
import streamlit as st
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

###### dotenv를 사용하지 않는 경우는 삭제하세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다.
사용 가능한 도구를 활용하여 조사한 정보를 설명해주세요.
이미 알고 있는 정보만으로 답변하지 말고, 가능한 한 검색을 수행한 뒤 답변해주세요.
(사용자가 읽을 페이지를 지정하는 등 특별한 경우는 검색하지 않아도 됩니다.)

검색 결과 페이지만 확인했을 때 정보가 충분하지 않다고 판단되면 다음 옵션을 고려해 시도해 주세요.

- 검색 결과의 링크를 클릭해 각 페이지의 콘텐츠를 열람하고 내용을 확인하세요.
- 한 페이지가 너무 길 경우, 3페이지 이상 스크롤하지 마세요 (메모리 부담 때문).
- 검색 쿼리를 변경한 뒤 다시 검색을 시도하세요.
- 검색할 주제에 따라 검색 언어를 적절하게 변경하세요.
  - 예: 프로그래밍 관련 질문은 영어로 검색하는 것이 더 유리할 수 있습니다.

사용자는 매우 바쁘며, 당신만큼 여유롭지 않습니다.
따라서 사용자의 수고를 덜어주기 위해 **직접적인 답변**을 제공해주세요.

=== 나쁜 답변 예시 ===
- 다음 페이지들을 참고하세요.
- 이 페이지들을 보고 코드를 작성할 수 있습니다.
- 다음 페이지가 도움이 될 것입니다.

=== 좋은 답변 예시 ===
- 이 문제의 해결 예시는 다음과 같습니다. -- 여기 코드 제시 --
- 질문에 대한 답은 다음과 같습니다. -- 여기 답변 제시 --

답변 마지막에는 **참조한 페이지의 URL을 반드시 기재**해주세요.
(사용자가 정보를 검증할 수 있도록)

사용자가 사용하는 언어로 답변해주세요.
사용자가 한국어로 질문하면 한국어로, 스페인어로 질문하면 스페인어로 답변해야 합니다.
"""


def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="🤗"
    )
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해주세요!"}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )
        # 아래와 같이도 작성할 수 있습니다
        # from langchain_community.chat_message_histories import StreamlitChatMessageHistory
        # msgs = StreamlitChatMessageHistory(key="special_app_key")
        # st.session_state['memory'] = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


def select_model():
    models = ("GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "GPT-3.5 (not recommended)")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5 (not recommended)":
        return ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=0, model_name="gpt-4o")
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=0, model_name="claude-3-5-sonnet-20240620")
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=0, model="gemini-1.5-pro-latest")


def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="2023 FIFA 여자 월드컵 우승 국가는?"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # 콜백 함수 설정 (에이전트 동작 시각화용)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # 에이전트 실행
            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
