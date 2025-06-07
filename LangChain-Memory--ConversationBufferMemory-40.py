from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 初始化对话记忆
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("你是谁？")
memory.chat_memory.add_ai_message("你好，我是LangChain专家。")

# 初始化对话链
conversation = ConversationChain(llm=llm, memory=memory)

# 进行对话
res = conversation.invoke({"input": "你是谁？"})
print(res)
