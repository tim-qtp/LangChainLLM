from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
load_dotenv()


# 初始化工具 可以根据需要进行配置
# 使用包装器WikipediaAPIWrapper进行搜索并获取页面摘要。默认情况下，它将返回前 k 个结果的页面摘要。它通过 doc_content_chars_max 限制文档内容。
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper) # API Wrapper 初始化了一个 WikipediaQueryRun 工具

# 工具默认名称
print("name:", tool.name)
# 工具默认的描述
print("description:", tool.description)
# 输入内容 默认JSON模式
print("args:", tool.args)
# 工具是否应该直接返回给用户
print("return_direct:", tool.return_direct) # 是否直接将工具结果返回给用户（一般为 False，表示交给 Agent 后处理）

# 可以用字典输入来调用这个工具
print(tool.run({"query": "langchain"}))
# 使用单个字符串输入来调用该工具。
print(tool.run("langchain"))

# {
#   'query': {
#     'description': 'query to look up on wikipedia',
#     'title': 'Query',
#     'type': 'string'
#   }
# }

# 🚀 意思如下：
# 字段	含义
# query	工具需要的参数名（键名）就是 "query"
# description	参数的含义是：要在维基百科中查询的关键词
# title	参数的标题，用于 UI 显示或文档说明
# type	参数类型是 string，也就是说你传的必须是字符串（比如 "langchain"）

from langchain.tools import tool

@tool
def add_number(a: int, b: int) -> int:
    """add two numbers."""
    return a + b

print(add_number.name)
print(add_number.description)
print(add_number.args)

res = add_number.run({"a": 10, "b": 20})
print(res)