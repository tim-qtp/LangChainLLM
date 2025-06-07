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

# 初始化Agent:使用工具、语言模型和代理类型来初始化代理    ZERO_SHOT_REACT_DESCRIPTION是一种不依赖预训练任务描述的 Zero-Shot ReAct Agent，会“观察输入问题 → 选择工具 → 执行动作 → 形成最终答案”。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# verbose=True：开启详细日志，可打印 Agent 的推理、工具选择、调用参数等。
# 让代理来回答提出的问题
agent.invoke({"input": "目前市场上苹果手机16的售价是多少？如果我在此基础上加价5%卖出，应该如何定价？用中文回答"})

# 1️⃣ Zero-Shot（零样本）
# 零样本表示：代理并没有看到类似的任务样本（或提示）来训练。
# 它也不依赖用户事先提供任务模板或任务类型。
# 它是“看都没看过类似问题”，但 靠语言模型自己的理解和推理能力，也能解决。

# 2️⃣ ReAct（Reasoning + Acting）
# 是一种智能体策略，整合了 推理（Reasoning） 和 行动（Acting）的过程。
# 即智能体先“想一想”，再“调用工具”，然后再“继续想”——这种策略比单纯问答更强。

# ReAct 策略在流程上长这样：
# 用户输入问题
#     ↓
# 模型思考（Reasoning）：我该怎么做？
#     ↓
# 模型选择一个工具（Act）
#     ↓
# 工具返回结果
#     ↓
# 模型继续思考（Reasoning）：结果意味着什么？
#     ↓
# 输出最终答案（Final Answer）

# ✅ 举个例子来形象说明：
# 用户输入：
# "目前 iPhone 16 的价格是多少？加价 5% 应该卖多少？"
# Agent 工作流程（Zero-Shot + ReAct）如下：
# 🤔 思考：我不知道价格，但可以查。
# 🔧 动作：调用 serpapi 搜索关键词 “iPhone 16 价格”。
# 📄 获取结果：例如“8999元”。
# 🤔 再思考：我需要计算加价 5%。
# 🔧 动作：调用 llm-math 工具计算 8999 × 1.05。
# ✅ 最终输出：结果是“9448.95元”。