import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

text = "大模型是什么？"
# llm = ChatOpenAI()

# llm = ChatOpenAI(
#     temperature=0.95,
#     model="glm-4",
#     openai_api_key="a224296ae6e44b8888e3853e9edee87e.2V9IraTGKBRFQRj6",
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
# )

# llm = ChatOpenAI(
#     model="moonshot-v1-8k",
#     openai_api_key="sk-BkIzEeSsz8gaJEo1VqjkqtJ01H4airZiQo98aZv5pPc9e4gn",
#     openai_api_base="https://api.moonshot.cn/v1"
# )

tongyi_api_key = os.getenv("TONGYI_API_KEY")

llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    openai_api_key=tongyi_api_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
res = llm.invoke(text)
res2 = llm.invoke("你是模型版本是什么？")

print(res.content)
print(res2.content)
