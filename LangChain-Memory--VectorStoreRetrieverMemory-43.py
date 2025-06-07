from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import faiss
# from langchain.docstore import InMemoryDocstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# 初始化向量存储
embedding_size = 1536  # OpenAIEmbeddings 的维度
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# 向量查找返回语义相关的信息
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
# 相关文档信息放到memory
memory = VectorStoreRetrieverMemory(retriever=retriever) # 这是 LangChain 中支持检索记忆的模块

# 当添加到代理中时，记忆对象可以保存对话中的重要信息或使用的工具
memory.save_context({"Human": "我最喜欢的食物是披萨"}, {"AI": "很高兴知道"})
memory.save_context({"Human": "我喜欢的运动是跑步"}, {"AI": "好的,我知道了"})
memory.save_context({"Human": "我最喜欢的运动是足球"}, {"AI": "好的,我知道了"})

llm = OpenAI(temperature=0)
_DEFAULT_TEMPLATE = """
以下是人类和人工智能之间的友好对话。人工智能很健谈，并从其上下文中提供了许多具体细节。如果人工智能不知道问题的答案，它就会如实说它不知道。

之前对话的相关片段: {history} 
(如果不相关，则无需使用这些信息)

当前对话:
Human: {input}
AI:
"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

# 创建Chain
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)

# 提问
res = conversation_with_summary.predict(input="我最喜欢的食物是什么呢？")
print(str(res))