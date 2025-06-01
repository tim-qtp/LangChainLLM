from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings_model = OpenAIEmbeddings()  # LangChain 封装的 OpenAI 文本嵌入（embedding）模型，默认使用 text-embedding-ada-002。

embeddings = embeddings_model.embed_documents(
    # embed_documents() 会将这些文本依次转换为嵌入向量。
    # 每个文本都会被编码为一个 1536 维的向量（text-embedding-ada-002 默认输出是 1536 维）。
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

print(len(embeddings), len(embeddings[0]))
print(embeddings[0][:5])
##运行结果 (5, 1536)


embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])