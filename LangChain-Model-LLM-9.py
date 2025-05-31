from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

text = "大模型是什么？"
llm = ChatOpenAI()

res = llm.invoke(text)
res2 = llm.invoke("你是模型版本是什么？")

print(res.content)
print(res2.content)
