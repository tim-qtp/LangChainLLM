
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain

# 初始化大模型
llm = ChatOpenAI(model="gpt-4o",temperature=0)

# 创建链
llm_math = LLMMathChain.from_llm(llm)

# 执行链
res = llm_math.invoke("10 ** 3 + 100的结果是多少？")
print(res)