from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [("system", """根据提供的上下文: {context} \n\n 回答问题: {input}""")]
)

# 初始化大模型
llm = ChatOpenAI(model="gpt-4o")

# 构建链  这个链将文档作为输入，并使用之前定义的提示模板和初始化的大模型来生成答案
chain = create_stuff_documents_chain(llm, prompt)

# 加载文档
loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
        )

docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
# print(documents)
print(len(documents))

# 执行链  检索  民事法律行为? 出来的结果
res = chain.invoke({"input": "民事法律行为?", "context": documents[:5]})
print(res)