# 导入和使用 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

# BeautifulSoup 是网页解析库，
import bs4


loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT")) # 只提取正文部分
    )

# 执行网页加载器，返回文档对象列表（LangChain 统一格式）
docs = loader.load()
# print(docs)

# 对于嵌入模型，这里通过 API调用
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

#使用此嵌入模型将文档摄取到矢量存储中
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用分割器分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
print(len(documents))
# 向量存储  embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 创建提示模板（Prompt），约定模型必须在 <context> 中找答案
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")
# 要求模型：只基于上下文回答问题，避免胡编。

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
# 创建文档组合链  将文档内容和用户问题组合成一个完整的提示，然后传递给语言模型生成回答
document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever() # 将 FAISS 向量库转为（加一层） Retriever 检索器，支持按语义相似度检索文档
retriever.search_kwargs = {"k": 3}  # 限制为最多检索3个文档
# 创建检索链   该链结合了检索器和文档组合链，实现了从向量数据库中检索相关文档，并将这些文档与用户问题组合成提示
retrieval_chain = create_retrieval_chain(retriever, document_chain) # 先查FAISS向量数据，添加到prompt，再发送llm
# 调用检索链并获取回答
response = retrieval_chain.invoke({"input": "建设用地使用权是什么？"}) # input放到prompt中
print(response["answer"])


# 设你网页中有这样 3 个文档片段：

# doc1: 建设用地使用权是指国家将国有土地出让给单位或个人用于建设的权利。
# doc2: 农用地转为建设用地须经严格审批。
# doc3: 使用权可出让、转让、出租，但不得违反规划。

# 你的问题 "建设用地使用权是什么？" 发送给 retriever 后，返回这 3 条上下文，然后会生成如下 prompt：


# 仅根据提供的上下文回答以下问题:

# <context>
# 建设用地使用权是指国家将国有土地出让给单位或个人用于建设的权利。
# 农用地转为建设用地须经严格审批。
# 使用权可出让、转让、出租，但不得违反规划。
# </context>

# 问题: 建设用地使用权是什么？

# 这个完整的提示就会被送入 GPT 模型中回答问题。