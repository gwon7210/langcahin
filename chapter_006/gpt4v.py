import base64
import streamlit as st
from langchain_openai import ChatOpenAI

###### dotenv 을 사용하지 않는 경우는 삭제하세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Image Recognizer",
        page_icon="🤗"
    )
    st.header("Image Recognizer 🤗")
    st.sidebar.title("Options")


def main():
    init_page()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        # 왜인지 max_tokens를 지정하지 않으면 동작이 안정적이지 않음 (2024년 2월 기준)
        # 매우 짧은 답변이 나오거나 중간에 끊기는 문제가 발생함.
        max_tokens=512
    )

    uploaded_file = st.file_uploader(
        label='이미지를 업로드해 주세요😇',
        # GPT-4V가 처리 가능한 이미지 파일만 허용
        type=['png', 'jpg', 'webp', 'gif']
    )
    if uploaded_file:
        if user_input := st.chat_input("물어보고 싶은 내용을 입력해 주세요!"):
            # 읽어온 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": user_input
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "auto"
                            },
                        }
                    ]
                )
            ]
            st.markdown("### 질문")
            st.write(user_input)     # 사용자의 질문
            st.image(uploaded_file)  # 업로드한 이미지 표시
            st.markdown("### 답변")
            st.write_stream(llm.stream(query))

    else:
        st.write('먼저 이미지를 업로드해 주세요😇')


if __name__ == '__main__':
    main()
