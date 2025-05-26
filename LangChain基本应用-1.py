from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI


# llm = ChatOpenAI()
llm = ChatOpenAI(model="gpt-4")


# 直接提供问题，并调用llm
# response = llm.invoke("什么是大模型？还有你的模型版本是多少?")
response = llm.invoke("什么是大模型？还有你的模型版本是多少?")
print(response)
print(response.content)