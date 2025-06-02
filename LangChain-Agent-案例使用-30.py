from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# 初始化大模型
# llm = ChatOpenAI(model="gpt-4", temperature=0)

llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    openai_api_key='sk-31bb7a65dd4047aba9b14a95c08be52c',
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 加载HTML内容为一个文档对象
loader = WebBaseLoader("https://new.qq.com/rain/a/20240920A07Y5Y00")
docs = loader.load()
# print(docs)

# 分割文档
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# 向量化
vector = FAISS.from_documents(documents, OpenAIEmbeddings())

# 创建检索器
retriever = vector.as_retriever()
# 创建一个工具来检索文档（这个就相当于拥有调用知识库的工具）
retriever_tool = create_retriever_tool(
    retriever, # 其实这个就相当于Dify中的知识库
    "iPhone_price_search",
    "搜索有关 iPhone 16 的价格信息。对于iPhone 16的任何问题，您必须使用此工具！",
)

# 加载所需的库
from langchain_community.tools.tavily_search import TavilySearchResults


# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()

# 创建将在下游使用的工具列表
tools = [search, retriever_tool] # 可以查询浏览器或者查询知识库

from langchain import hub

# 这是从 LangChain Hub 拉取一个标准的 Agent Prompt 模板，这个 Prompt 会告诉大模型：
# “你是一个智能助手，可以调用下列工具（例如：搜索工具、知识库检索工具），请根据用户问题决定是否需要调用工具、调用哪个工具、如何调用，并最终输出答案。”
# 比如模板里会有：
# You are an agent designed to answer questions by calling tools.
# You have access to the following tools:

prompt = hub.pull("hwchase17/openai-functions-agent")
# 打印Prompt
# print(prompt)

# 使用OpenAI functions代理
from langchain.agents import create_openai_functions_agent

# 构建OpenAI函数代理：使用 LLM、提示模板和工具来初始化代理
agent = create_openai_functions_agent(llm, tools, prompt) # 根据用户问题，判断agent需要调用以下哪些工具

from langchain.agents import AgentExecutor
# 将代理与AgentExecutor工具结合起来
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # 这个是真的在调用

# 执行代理
agent_executor.invoke({"input": "目前市场上苹果手机16的各个型号的售价是多少？"})
#agent_executor.invoke({"input": "美国2024年谁胜出了总统的选举?"})

# 这里不太懂过为什么为什么还要再传一遍 tools
# Agent 是“我想用工具A”，Executor 是“我真把工具A调用了”。
# 类似以下这样：
# | 元素               | 比喻对象          | 含义                                                             |
# | ----------------  | -----------      | -------------------------------------------                     |
# | `agent`           | 厨师              | 会根据菜单（prompt）和顾客问题（input）决定“我需要用什么食材”（哪个工具） |
# | `tools`           | 食材清单           | 比如：西红柿、鸡蛋、青椒、牛肉（FAISS、搜索引擎等工具）                 |
# | `agent_executor` |  点菜员 + 厨房调度员 | 真正让厨师开始做菜、调动食材的人，是执行中枢                           |
