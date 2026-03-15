# ssh -T git@github.com
# streamlit run app.py --server.port 8080
# !pip install langchain==0.1.14
# !pip install langchain-openai==0.0.8
# !pip install langchainhub duckduckgo-search wikipedia
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler
# langchain==0.3.28
# from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
# langchain==0.1.14
from langchain.schema import HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
# from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
import streamlit as st
import os


load_dotenv()


def create_agent_chain(history):
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"]
    )
    # tools = load_tools(["terminal"], allow_dangerous_tools=True)
    tools = load_tools(["ddg-search", "wikipedia"])
    # prompt = hub.pull("hwchase17/react")
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt = hub.pull("hwchase17/openai-tools-agent")

    memory = ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=True)

    # agent = create_react_agent(llm, tools, prompt)
    # agent = create_openai_functions_agent(llm, tools, prompt)
    agent = create_openai_tools_agent(chat, tools, prompt)
    # return AgentExecutor(agent=agent, tools=tools)
    # return AgentExecutor(agent=agent, tools=tools, verbose=True)
    # conversation = ConversationChain(llm=chat, memory=ConversationBufferMemory())
    return AgentExecutor(agent=agent, tools=tools, memory=memory)


st.title("lanchain-streamlit-app")
history = StreamlitChatMessageHistory()
for message in history.messages:
    st.chat_message(message.type).write(message.content)

prompt = st.chat_input("What is up?")
# print(prompt)
if prompt:
    with st.chat_message("user"):
        # history.add_user_message(prompt)
        # history.add_user_message(HumanMessage(content=prompt))
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # response = "안녕하세요"
        # chat = ChatOpenAI(
        #     model_name=os.environ["OPENAI_API_MODEL"],
        #     temperature=os.environ["OPENAI_API_TEMPERATURE"]
        # )
        callback = StreamlitCallbackHandler(st.container())
        # agent_chain = create_agent_chain()
        agent_chain = create_agent_chain(history)
        # messages = [HumanMessage(content=prompt)]
        # response = chat.invoke(messages)
        # response = chat.invoke(history.messages)
        # response = agent_chain.invoke(
        #     {"input": prompt},
        #     {"callback": [callback]}
        # )
        response = agent_chain.invoke(
            {"input": prompt, "chat_history": history.messages},
            {"callbacks": [callback]}
        )
        # history.add_ai_message(response)
        # history.add_ai_message(response["output"])
        # st.markdown(response)
        # st.markdown(response.content)
        st.markdown(response["output"])
