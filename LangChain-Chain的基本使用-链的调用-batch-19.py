from dotenv import load_dotenv
load_dotenv()
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 创建模型实例
template = PromptTemplate(
    input_variables=["role", "fruit"],
    template="{role}喜欢吃{fruit}?",
)

# 创建LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建LLMChain
# llm_chain = LLMChain(llm=llm, prompt=template)
llm_chain = template | llm

# 输入列表
input_list = [
    {"role": "猪八戒", "fruit": "人参果"}, {"role": "孙悟空", "fruit": "仙桃"}
]

# 调用LLMChain，返回结果
result = llm_chain.batch(input_list)
print(result)