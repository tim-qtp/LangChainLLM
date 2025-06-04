from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import langchain

# 开启DEBUG 显示具体的日志信息
# langchain.debug = True
# langchain.verbose = True

# 初始化大模型:语言模型控制代理
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 设置工具:加载使用的工具，serpapi:调用Google搜索引擎 llm-math:通过LLM进行数学计算的工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent:使用工具、语言模型和代理类型来初始化代理    ZERO_SHOT_REACT_DESCRIPTION 类型的代理可以在没有预先训练的情况下尝试解决新的问题
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 让代理来回答提出的问题
agent.invoke({"input": "目前市场上苹果手机16的售价是多少？如果我在此基础上加价5%卖出，应该如何定价？用中文回答"})