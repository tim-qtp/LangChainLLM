from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("qtp.pdf")
pages = loader.load_and_split()

print(f"第0页：\n{pages[8]}") ## 也可通过 pages[0].page_content只获取本页内容