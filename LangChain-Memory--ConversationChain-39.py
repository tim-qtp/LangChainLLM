# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from dotenv import load_dotenv
load_dotenv()


# 初始化大语言模型
llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-4o"
)

# 初始化对话链
conv_chain = ConversationChain(llm=llm)

# 打印对话的模板
print(conv_chain.prompt.template)
res = conv_chain.invoke("我是波波老师？")
print(res)
res = conv_chain.invoke("你是谁")
print(res)
res = conv_chain.invoke("我是谁?")
print(res)