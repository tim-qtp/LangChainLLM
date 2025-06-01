## 1. 文档加载
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("./qtp.pdf")
pages = loader.load_and_split()

## 2. 文档切分
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
paragraphs = []
for page in pages:
    paragraphs.extend(text_splitter.create_documents([page.page_content]))

# print(paragraphs)
## 3. 文档向量化，向量数据库存储
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(paragraphs, OpenAIEmbeddings())

## 4. 向量检索
# retriever = db.as_retriever()
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke("查看TCP监听端口和进程信息")
for doc in docs:
    print(f"{doc.page_content}\n-------\n")

## 5. 组装Prompt模板
import os
# 加载 .env 到环境变量
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o") # 默认是gpt-3.5-turbo

prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{info}

用户问：
{question}

请用中文回答用户问题。
"""

from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(prompt_template)

prompt = template.format(info=docs[0].page_content, question='怎么查看TCP监听端口和进程信息？')
print(prompt)
## 6. 调用LLM
response = llm.invoke(prompt)
print(response.content)