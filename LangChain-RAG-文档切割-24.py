from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./qtp.pdf")
pages = loader.load_and_split()
# print(f"第0页：\n{pages[0].page_content}")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,           # 每个文本块的最大长度
    chunk_overlap=100,        # 每个文本块之间的重叠字符数
    length_function=len,      # 使用 len() 计算长度
    add_start_index=True,     # 给每个切分块记录起始索引
)

paragraphs = text_splitter.create_documents([pages[0].page_content])
for para in paragraphs:
    print(para.page_content)
    print('-------')