from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# 创建模型实例
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"
prompt = PromptTemplate(template=template, input_variables=["number"])

# 创建LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
# 创建LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)
# llm_chain = prompt | llm
# 调用LLMChain，返回结果
result = llm_chain.predict(number=3)
print(result)