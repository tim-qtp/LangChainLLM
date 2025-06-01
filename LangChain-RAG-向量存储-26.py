from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader("./qtp.pdf")
pages = loader.load_and_split()

## 2. 文档切分
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter( # 使用 RecursiveCharacterTextSplitter 将每页文本切成 200 字符的段落，段与段之间重叠 100 字符。
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
# 最终的 paragraphs 是多个 Document 对象的小段集合，为后续向量化准备。
paragraphs = [] # 其实最后是一个二维数组（第一页的分段，第二页的分段...）
for page in pages:
    paragraphs.extend(text_splitter.create_documents([page.page_content]))

# print(paragraphs)
## 3. 文档向量化，向量数据库存储
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(paragraphs, OpenAIEmbeddings()) # 将文档的嵌入向量存入内存中的向量数据库 Chroma 中，便于快速检索

# query = "llama2有多少参数？"
# query = "怎么启动Mongo？"
query = "基础脚⼿架代码？"

docs = db.similarity_search(query) ## 一行代码搞定
for doc in docs:
    print(f"{doc.page_content}\n-------\n")
