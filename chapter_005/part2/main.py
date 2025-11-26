# Github: https://github.com/naotaka1128/llm_app_codes/chapter05/part2/main.py

import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from urllib.parse import urlparse
from langchain_community.document_loaders import YoutubeLoader  # Youtube용

###### dotenv를 사용하지 않는 경우 삭제하세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


SUMMARIZE_PROMPT = """다음 콘텐츠의 내용을 약 300자 정도로 알기 쉽게 요약해주세요.

========

{content}

========

한국어로 작성해 주세요!
"""


def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
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


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", SUMMARIZE_PROMPT),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def validate_url(url):
    """ URL이 유효한지 판단하는 함수 """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    """
    Document:
        - page_content: str
        - metadata: dict
            - source: str
            - title: str (add_video_info=False 일 경우 없음)
            - description: Optional[str],
            - view_count: int
            - thumbnail_url: Optional[str]
            - publish_date: str
            - length: int
            - author: str
    """
    with st.spinner("Fetching Youtube ..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,  # 중요: 에러 방지를 위해 메타데이터 가져오기 끔
                language=['ko', 'en']  # 한국어 우선, 없으면 영어
            )
            res = loader.load()  # list of `Document` (page_content, metadata)
            
            if res:
                content = res[0].page_content
                # [수정됨] title이 없으면 "YouTube Video"를 기본값으로 사용
                title = res[0].metadata.get('title', "YouTube Video")
                return f"Title: {title}\n\n{content}"
            else:
                return None

        except Exception as e:
            # 에러 발생 시 화면에 출력하여 디버깅 용이하게 함
            st.error(f"Error occurred: {e}")
            st.write(traceback.format_exc()) 
            return None


def main():
    init_page()
    chain = init_chain()

    # 사용자의 입력을 감시
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

    # 비용을 표시하려면 3장과 동일한 구현을 추가하세요
    # calc_and_display_costs()


if __name__ == '__main__':
    main()