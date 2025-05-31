from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_openai import OpenAIEmbeddings

# 初始化 OpenAIEmbeddings 实例
# embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 定义一个文本字符串
text = "大模型"

# 嵌入文档
doc_result = embeddings.embed_documents([text])
# 输出嵌入向量的前5个维度
print(doc_result[0][:5])
print(len(doc_result[0]))

# 嵌入查询
query_result = embeddings.embed_query(text)
print(doc_result[0][:5])
print(query_result[:5])
