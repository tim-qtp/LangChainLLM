from dotenv import load_dotenv
load_dotenv()
from langchain.prompts.chat import ChatPromptTemplate

# template = "你是一个数学家，你可以计算任何算式"
template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
# print(chat_prompt)
# exit()
# 导入LangChain中的ChatOpenAI模型接口
from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(model='gpt-4')
# 输入提示
# messages = chat_prompt.format_messages(text="我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？")
# print(messages)
messages = chat_prompt.format_messages(input_language="英文", output_language="中文", text="I love Large Language Model.")
print(messages)
# 得到模型的输出
output = model.invoke(messages)
# 打印输出内容
print(output.content)


# ==========================================
print("="*50)


# 导入聊天消息类模板
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 模板的构建
system_template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 格式化提示消息生成提示
prompt = prompt_template.format_prompt(input_language="英文", output_language="中文",
                                       text="I love Large Language Model.").to_messages()
# 打印模版
print("prompt:", prompt)
from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(model='gpt-3.5-turbo')
# 得到模型的输出
result = model.invoke(prompt)
# 打印输出内容
print("result:", result.content)