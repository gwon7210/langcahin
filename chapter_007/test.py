아래 배경 지식을 사용해서 사용자의 질문에 대답해주세요

===

배경 지식 - 1
{context_a}

===

배경 지식 - 2
{context_b}

===
사용자의 질문
{question}
"""
)

chain = (
    {
        "context_a": itemgetter("keyword") | retriever_a,
        "context_b": itemgetter("keyword") | retriever_b,
        "question": itemgetter("question"),
    }
    | prompt
)

chain.invoke({
    "keyword": "해외 사업 상황",
    "question": "이 두 기업의 결산 자료를 비교해주세요"
})