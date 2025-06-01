# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
#import os

#os.environ["http_proxy"] = "http://127.0.0.1:7897"
#os.environ["https_proxy"] = "http://127.0.0.1:7897"

# 原始字符串模板
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"

# 创建LangChain模板
prompt_temp = PromptTemplate.from_template(template)

# 根据模板创建提示
prompt = prompt_temp.format(number=2)

# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(model="gpt-4",temperature=0)
# 传入提示，调用模型返回结果
result = model.invoke(prompt)
print(result)