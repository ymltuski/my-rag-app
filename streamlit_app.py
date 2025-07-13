import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import os
import sys

# 自定义模块路径（请确认路径存在）
sys.path.append(os.path.abspath("/workspaces/test_codespace/A_Test/A_test/my_modules"))

from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- 工具函数 ----------
def get_retriever():
    embedding = ZhipuAIEmbeddings()
    persist_directory = '/workspaces/test_codespace/A_Test/A_test/my_modules/chroma_db'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。请使用检索到的上下文片段回答这个问题。如果你不知道答案就说不知道。"
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return RunnablePassthrough().assign(context=retrieve_docs).assign(answer=qa_chain)

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res:
            yield res["answer"]

# ---------- 主界面 ----------
def main():
    st.markdown("### 🦜🔗 动手学大模型应用开发")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()

    messages = st.container(height=550)

    for role, content in st.session_state.messages:
        with messages.chat_message(role):
            st.write(content)

    if prompt := st.chat_input("请输入你的问题"):
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)

        with messages.chat_message("ai"):
            full_response = st.write_stream(
                gen_response(
                    chain=st.session_state.qa_history_chain,
                    input=prompt,
                    chat_history=st.session_state.messages[:-1]
                )
            )
        st.session_state.messages.append(("ai", full_response))

if __name__ == "__main__":
    main()
