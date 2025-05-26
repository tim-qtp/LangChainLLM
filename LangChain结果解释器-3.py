from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 初始化模型
# llm = ChatOpenAI()
# 它会自动从 .env 读取你的 OPENAI_API_KEY，或者你可以手动指定：llm = ChatOpenAI(api_key="sk-xxx")
llm = ChatOpenAI(model="gpt-4")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "您是世界级的技术文档编写者。"),
    ("user", "{input}")
])

# 使用输出解析器
output_parser = StrOutputParser() # 输出的是字符串
# output_parser = JsonOutputParser() # 输出的是字典

# 将其添加到上一个链中
chain =  prompt | llm | output_parser
#chain = prompt | llm

# 调用它并提出同样的问题。答案是一个字符串，而不是ChatMessage
# chain.invoke({"input": "LangChain是什么?"})
response = chain.invoke({"input": "LangChain是什么? 问题用question 回答用answer"})
# response = chain.invoke({"input": "LangChain是什么? 问题用question 回答用answer 用JSON格式回复"})
print(response)
print(type(response))


# LangChain 的链式写法（|）是个 “数据流管道”：
# prompt 模板收到 input
# 转换成 OpenAI 消息格式传给模型
# 模型生成回应（期望是 JSON 格式）
# 输出解析器将模型结果变成结构化数据（Python 对象）

