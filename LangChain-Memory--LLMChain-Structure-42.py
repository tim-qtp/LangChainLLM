from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()



# 如果使用聊天模型，使用结构化的聊天消息可能会有更好的性能。
# 初始化大模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 使用ChatPromptTemplate设置聊天提示
prompt = ChatPromptTemplate.from_messages( # （更适合 chat 模型）
    [
        SystemMessage(content="你是一个与人类对话的机器人。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
# print(prompt)

# 创建ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化链
chain = LLMChain(llm=llm,  prompt=prompt, memory=memory)

# 提问
res = chain.invoke({"question": "你是LangChain专家"})
print(str(res) + "\n")

res = chain.invoke({"question": "你是谁?"})
print(res)


# 🧪 哪个更强？
# ChatPromptTemplate + ChatOpenAI 组合：
# 支持多轮对话结构化记忆
# 能区分 system、user、assistant 角色
# 响应更自然、更稳定
# 是未来主流推荐方式