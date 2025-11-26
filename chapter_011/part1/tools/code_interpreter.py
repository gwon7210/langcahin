# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_011/part1/tools/code_interpreter.py

import streamlit as st
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ExecPythonInput(BaseModel):
    """ 타입을 지정하기 위한 클래스 """
    code: str = Field()


@tool(args_schema=ExecPythonInput)
def code_interpreter_tool(code):
    """
    Code Interpreter를 사용하여 Python 코드를 실행합니다.
    - 아래와 같은 작업에 적합합니다.
      - pandas나 matplotlib 등의 라이브러리를 사용하여 데이터 가공 및 시각화를 수행할 수 있습니다.
      - 수식 계산이나 통계적 분석도 수행할 수 있습니다.
      - 자연어 처리 라이브러리를 사용하여 텍스트 데이터 분석도 가능합니다.
    - Code Interpreter는 인터넷에 연결할 수 없습니다
      - 외부 웹사이트의 정보를 읽거나 새로운 라이브러리를 설치할 수 없습니다
    - Code Interpreter가 작성한 코드를 함께 출력하도록 요구하면 좋습니다
      - 사용자가 결과를 검증하기 쉬워집니다
    - 어느 정도 코드가 틀려 있어도 자동으로 수정해줄 때가 있습니다

    Returns:
    - text: Code Interpreter가 출력한 텍스트 (코드 실행 결과)
    - files: Code Interpreter가 저장한 파일의 경로
        - 파일은 `./files/` 아래에 저장됩니다.
    """
    return st.session_state.code_interpreter_client.run(code)

