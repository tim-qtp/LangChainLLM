import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage,SystemMessage

text = "你好啊"
# 聊天模型
chat_model = ChatOpenAI(model="gpt-4o")
messages = [HumanMessage(content=text)]

res = chat_model.invoke(messages)
print(res)