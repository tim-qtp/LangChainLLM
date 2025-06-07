from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# 初始化大模型
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# 创建提示   有两个输入键：实际输入与来自记忆类的输入 需确保PromptTemplate和ConversationBufferMemory中的键匹配
# {chat_history}：表示之前的对话内容（由 memory 提供）
# {question}：表示用户当前提的问题
template = """你可以与人类对话。
当前对话: {chat_history}
人类问题: {question}
回复:
"""

# 这个模板中需要两个变量：chat_history 和 question
# 使用上面写的 template 作为文本格式
prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=template
)

# 这是一个对话记忆类，会自动将每次用户和 AI 的问答记录下来，并提供给下一次调用。注意这里的键 chat_history 要与 PromptTemplate 一致。
# 创建一个对话缓存器，保存历史问答内容
# 把历史内容放入变量名 chat_history 中，以便插入 PromptTemplate 的 {chat_history} 位置
memory = ConversationBufferMemory(memory_key="chat_history")

# 初始化链
chain = LLMChain(llm=llm,  prompt=prompt, memory=memory) #每次调用 .invoke() 就会自动加入历史记录。

# 提问
# 第一轮对话，用户问：“你是LangChain专家吗？”
# 此时 chat_history 是空的
# 模型只看到当前问题
res = chain.invoke({"question": "你是LangChain专家"})
print(str(res) + "\n")


res1 = chain.invoke({"question": "我的名字叫波波老师"})
print(str(res1) + "\n")

res = chain.invoke({"question": "我的名字是什么?"})
print(res)