from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI


llm = ChatOpenAI()
# llm = ChatOpenAI(model_name="gpt-4")

# 我们也可以创建prompt template, 并引入一些变量到prompt template中，这样在应用的时候更加灵活
from langchain_core.prompts import ChatPromptTemplate

# 需要注意的一点是，这里需要指明具体的role，在这里是system和用户
prompt = ChatPromptTemplate.from_messages([
    ("system", "您是世界级的技术文档编写者"),
    ("user", "{input}")  # {input}为变量
])

# 我们可以把prompt和具体llm的调用和在一起（通过chain，chain可以理解为sequence of calls to take）  Linux  ps aux | grep redis 管道符号
chain = prompt | llm
response = chain.invoke({"input": "大模型中的LangChain是什么?"})
print(response)
print(response.content)