import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

###### dotenv 을 사용하지 않는 경우는 삭제하세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


GPT4V_PROMPT = """
먼저, 아래의 사용자의 요청과 업로드된 이미지를 주의 깊게 읽어주세요.

다음으로, 업로드된 이미지를 기반으로 이미지를 생성해 달라는 사용자의 요청에 따라
DALL-E용 프롬프트를 작성해 주세요.
DALL-E 프롬프트는 반드시 영어로 작성해야 합니다.

주의: 이미지 속 사람이나 특정 장소, 랜드마크, 상표 등을 식별하지 말아 주세요.
묘사는 사진 속 시각적 요소를 중립적으로 설명하는 방식으로 해주세요.

사용자 입력: {user_input}

프롬프트에서는 사용자가 업로드한 사진에 무엇이 담겨 있는지,
어떻게 구성되어 있는지를 설명해 주세요.
사진의 구도와 줌 정도도 설명해 주세요.
사진의 내용을 재현하는 것이 중요합니다.

DALL-E 3용 프롬프트를 영어로 출력해 주세요:
"""



def init_page():
    st.set_page_config(
        page_title="Image Converter",
        page_icon="🤗"
    )
    st.header("Image Converter 🤗")


def main():
    init_page()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        # 이유는 알 수 없지만 max_tokens가 없으면 가끔 동작이 불안정해짐 (2024년 5월 기준)
        max_tokens=512
    )

    dalle3_image_url = None
    uploaded_file = st.file_uploader(
        label='이미지를 업로드해 주세요😇',
        # GPT-4V가 처리 가능한 이미지 파일만 허용
        type=['png', 'jpg', 'webp', 'gif']
    )
    if uploaded_file:
        if user_input := st.chat_input("이미지를 어떻게 가공하고 싶은지 알려줘!"):
            # 읽은 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": GPT4V_PROMPT.format(user_input=user_input)
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

            # GPT-4V에게 DALL-E 3용 이미지 생성 프롬프트를 작성하게 함
            st.markdown("### Image Prompt")
            image_prompt = st.write_stream(llm.stream(query))

            # DALL-E 3로 이미지 생성
            with st.spinner("DALL-E 3가 그림을 그리는 중입니다..."):
                dalle3 = DallEAPIWrapper(
                    model="dall-e-3",
                    size="1792x1024",   # "1024x1024", "1024x1792"도 선택 가능
                    quality="standard", # 'hd'로 더 고품질 이미지 생성 가능
                    n=1,  # 한 번에 1장만 생성 가능
                )
                dalle3_image_url = dalle3.run(image_prompt)
    else:
        st.write('먼저 이미지를 업로드해 주세요😇')

    # DALL-E 3 결과 이미지 표시
    if dalle3_image_url:
        st.markdown("### Question")
        st.write(user_input)
        st.image(
            uploaded_file,
            use_column_width="auto"
        )

        st.markdown("### DALL-E 3 Generated Image")
        st.image(
            dalle3_image_url,
            caption=image_prompt,
            use_column_width="auto"
        )


if __name__ == '__main__':
    main()


#✅ 문제 정의
#
#한국어로 이미지 변환 요청을 입력할 때, GPT-4o(Vision)가 간헐적으로
#**“특정 장소나 랜드마크를 식별할 수 없다”**는 안전성 경고 메시지를 출력하며
#DALL-E용 프롬프트 생성이 실패하는 현상이 발생했다.
#같은 요청을 영어로 입력하면 정상 동작하는 불규칙성이 있었다.
#
#✅ 원인
#
#한국어 명령문이 Vision 모델에 의해 장소·인물 식별 요청으로 잘못 판단되어
#GPT-4o의 이미지 안전성 정책이 발동한 것.
#특히 기존 프롬프트 내용이 Vision 정책과 충돌하는 방식으로 해석될 여지가 있어
#한국어 입력에서 오류가 더 빈번하게 나타났다.
#
#✅ 해결 방안
#
#프롬프트에 **“특정 장소·인물·랜드마크·상표 등을 식별하지 말라”**는 안전 지침을 명시하고,
#이미지 설명을 중립적·비식별적 묘사 중심으로 완화하도록 수정했다.
#이로 인해 Vision 정책 충돌이 사라지고 한국어 입력에서도 일관되고 안정적으로
#DALL-E용 프롬프트가 생성되도록 개선되었다.


#🌟 추가 수정 필요 사항
# DallEAPIWrapper에서는 dall-e-3만 사용 가능하며 다른 모델은 지원하지 않는다.
# gpt-image-1 같은 최신 이미지 모델을 사용하려면 OpenAI Images API 또는 LangChain OpenAIImage로 코드를 변경해야 한다.
# https://platform.openai.com/docs/api-reference/images/create?utm_source=chatgpt.com 여기 참고해서 각 파라미터 의미도 추가 설명 필요