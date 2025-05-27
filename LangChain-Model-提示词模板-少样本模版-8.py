from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate

examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]

# 创建提示模板，配置一个提示模板，将一个示例格式化为字符串
prompt_template = "你是一个数学专家,算式： {input} 值： {output} 使用： {description} "

# 这是一个提示模板，用于设置每个示例的格式
prompt_sample = PromptTemplate.from_template(prompt_template)

# input="4+2", output="6", description="加法运算"
print(prompt_sample.format_prompt(**examples[0]))
#print(prompt_sample.format_prompt(input="2+2", output="4", description="加法运算"))


# 创建一个FewShotPromptTemplate对象
from langchain.prompts.few_shot import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=examples, # 这两个就是提示作用
    example_prompt=prompt_sample, # 这两个就是提示作用，把例子放到模版中，形成真正的少量提示词
    suffix="你是一个数学专家,算式: {input}  值: {output}", # 这个就是真正的提示词了！
    input_variables=["input", "output"]
)
print(prompt.format(input="2*5", output="10"))  # 你是一个数学专家,算式: 2*5  值: 10


from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4')
result = model.invoke(prompt.format(input="2*5", output="10"))
print(result.content)  # 使用: 乘法运算