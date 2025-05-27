# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate

# 创建原始模板
template = "您是一位专业的程序员。\n对于信息 {text} 进行简短描述"

# 根据原始模板创建LangChain提示模板
prompt_template1 = PromptTemplate.from_template(template)

# 打印LangChain提示模板的内容
print(prompt_template1)
print("="*50)
# ==================================================
print(prompt_template1.format(text="langchain"))
print("="*50)
# ==================================================
prompt_template2 = PromptTemplate(
    input_variables=["text"],
    template="您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
)
print(prompt_template2.format(text="langchain"))
print("="*50)
# ==================================================

# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# 创建模型实例
model = ChatOpenAI(model='gpt-4')

prompt = PromptTemplate(
    input_variables=["text"],
    template="您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
)
# 输入提示
input = prompt.format(text="大模型 langchain")

print(input)
print("="*50)
# ==================================================
# 得到模型的输出
output = model.invoke(input)
# output = model.invoke("您是一位专业的程序员。对于信息 langchain 进行简短描述")

# 打印输出内容
print(output.content)