import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### dotenv を利用しない場合は消してください ######
# dotenv를 사용하지 않는 경우는 삭제하세요
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🧐"
    )
    st.sidebar.title("옵션")


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("모델을 선택하세요:", models)
    if model == "GPT-3.5":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo"
        )
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o"
        )
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature,
            model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-1.5-pro-latest"
        )


def init_qa_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template("""
    아래의 전제 지식을 바탕으로 사용자 질문에 답변해 주세요.

    ===
    전제 지식
    {context}

    ===
    사용자 질문
    {question}
    """)
    retriever = st.session_state.vectorstore.as_retriever(
        # "mmr", "similarity_score_threshold" 등도 사용 가능
        search_type="similarity",
        # 몇 개의 문서를 가져올지 (기본값: 4)
        search_kwargs={"k": 10}
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def page_ask_my_pdf():
    chain = init_qa_chain()

    if query := st.text_input("PDF에 대한 질문을 입력하세요: ", key="input"):
        st.markdown("## 답변")
        st.write_stream(chain.stream(query))


def main():
    init_page()
    st.title("PDF QA 🧐")
    if "vectorstore" not in st.session_state:
        st.warning("먼저 📄 Upload PDF(s)에서 PDF 파일을 업로드해 주세요")
    else:
        page_ask_my_pdf()


if __name__ == '__main__':
    main()
