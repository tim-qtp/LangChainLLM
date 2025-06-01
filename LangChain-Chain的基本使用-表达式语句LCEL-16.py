from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

template = """Write some python code to solve the user's problem.

Return only python code in Markdown format, e.g.:

​```python
....
​```"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = ChatOpenAI(model="gpt-4")

def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

# PythonREPL().run 就是调用了一下 exec 函数执行代码
# chain = prompt | model | StrOutputParser()
chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run
result = chain.invoke({"input": "whats 2 plus 2"})

print(result)