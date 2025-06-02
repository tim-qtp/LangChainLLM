# 加载所需的库
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()

# 创建将在下游使用的工具列表
tools = [search]

from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    openai_api_key='sk-31bb7a65dd4047aba9b14a95c08be52c',
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


from langchain import hub

# 获取要使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")

# 使用OpenAI functions代理
from langchain.agents import create_openai_functions_agent

# 创建使用 OpenAI 函数调用的代理
agent = create_openai_functions_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

# 得到代理工具执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 表示会在终端输出运行过程
# 执行代理
# 传入一个空的消息列表给chat_history，因为它是聊天中的第一条消息
from langchain_core.messages import AIMessage, HumanMessage
chat_history = []
input = "hello 我是QTP老师"

res = agent_executor.invoke({"input": input, "chat_history": chat_history})
# print("res", res)
#
chat_history.append(HumanMessage(content=res['input']))
chat_history.append(AIMessage(content=res['output']))

# input = "你是LangChain专家"
#
# res1 = agent_executor.invoke({"input": input, "chat_history": chat_history})
# print(res1)
# print('='*10)
# chat_history.append(HumanMessage(content=input))
# chat_history.append(AIMessage(content=res1['output']))

# print("chat_history", chat_history)
# print('='*10)

agent_executor.invoke(
    {
        "input": "我是谁?",
        "chat_history": chat_history
    }
)