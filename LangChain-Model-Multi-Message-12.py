import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage,SystemMessage

text = "你好啊"
# 聊天模型
chat_model = ChatOpenAI(model="gpt-4o")
messages = [
    SystemMessage(content="你是一位乐于助人的助手。你叫波波老师"),
    HumanMessage(content="你是谁？")
]

res = chat_model.invoke(messages)
print(res)