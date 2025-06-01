from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# 原始字符串模板
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"

# 创建模型实例
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建LLMChain
# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate.from_template(template)
# )

# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0.
# Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
# llm_chain = LLMChain(
prompt=PromptTemplate.from_template(template)
llm_chain = prompt | llm

# 调用LLMChain，返回结果
result = llm_chain.invoke({"number":2})
print(result)