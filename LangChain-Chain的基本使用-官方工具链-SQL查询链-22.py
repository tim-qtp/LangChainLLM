from langchain_community.utilities import SQLDatabase

# 连接 sqlite 数据库
# db = SQLDatabase.from_uri("sqlite:///demo.db")

# 连接 MySQL 数据库
db_user = "root"
db_password = "123.com.."
db_host = "127.0.0.1"
db_port = "3306"
db_name = "dify-qtp"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

print("那种数据库：",db.dialect)
print("获取数据表：",db.get_usable_table_names())
# 执行查询
res = db.run("SELECT count(*) FROM goods;")
print("查询结果：",res)



from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from dotenv import load_dotenv
load_dotenv()
# 初始化大模型
llm = ChatOpenAI(model="gpt-4", temperature=0)

chain = create_sql_query_chain(llm=llm, db=db)

#response = chain.invoke({"question": "查询考试科目一共有多少？"})
# response = chain.invoke({"question": "查询一班的学生数学成绩是多少？"})
response = chain.invoke({"question": "查询小米电视的价格？"})
response2 = chain.invoke({"question": "查询商品表中有几个字段？"})
response3 = chain.invoke({"question": "查询消费者表中的所有消费者名字，姓名拼在一起，不要分页返回？"})
# 限制使用的表
response4 = chain.invoke({"question": "一共有多少个品牌？", "table_names_to_use": ["vendors"]})
print(response)
print(response2)
print(response3)
print(response4)