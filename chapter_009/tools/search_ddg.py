# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/tools/search_ddg.py

from itertools import islice
from duckduckgo_search import DDGS
from langchain_core.tools import tool
# from langchain_core.pydantic_v1 import (BaseModel, Field)
from pydantic import BaseModel, Field


"""
DuckDuckGo Python 라이브러리의 응답 예시
--------------------------------------------
[
    {
        'title': '일정·결과｜FIFA 여자 월드컵 호주 & 뉴질랜드 2023｜나데시코 재팬｜일본 대표｜JFA｜일본축구협회',
        'href': 'https://www.jfa.jp/nadeshikojapan/womensworldcup2023/schedule_result/',
        'body': '일정·결과｜FIFA 여자 월드컵 호주 & 뉴질랜드 2023｜나데시코 재팬｜일본 대표｜JFA｜일본축구협회. FIFA 여자 월드컵. 호주 & 뉴질랜드 2023.'
    }, ...
]
"""

class SearchDDGInput(BaseModel):
    query: str = Field(description="검색할 키워드를 입력하세요")


@tool(args_schema=SearchDDGInput)
def search_ddg(query, max_result_num=5):
    """
    DuckDuckGo 검색을 실행하는 도구입니다.
    검색할 키워드를 입력해 사용해 주세요.
    검색 결과의 각 페이지에 대한 제목, 스니펫(설명), URL이 반환됩니다.
    이 도구에서 얻을 수 있는 정보는 매우 단순화되어 있으며, 경우에 따라 오래된 정보일 수도 있습니다.

    원하는 정보를 찾지 못했다면 반드시 `WEB Page Fetcher` 도구를 사용해 각 페이지의 실제 내용을 확인해 주세요.
    문맥에 따라 가장 적합한 언어로 검색하세요 (사용자의 언어와 같을 필요는 없습니다).
    예를 들어, 프로그래밍 관련 질문이라면 영어로 검색하는 것이 가장 효과적입니다.

    Returns
    -------
    List[Dict[str, str]]:
    - title
    - snippet
    - url
    """
    res = DDGS().text(query, region='wt-wt', safesearch='off', backend="lite")
    return [
        {
            "title": r.get('title', ""),
            "snippet": r.get('body', ""),
            "url": r.get('href', "")
        }
        for r in islice(res, max_result_num)
    ]
