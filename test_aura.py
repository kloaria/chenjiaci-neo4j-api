import os
from neo4j import GraphDatabase

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print("URI =", uri)
print("USERNAME =", username)

driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)

try:
    driver.verify_connectivity()

    print("✅ Neo4j Aura 连接成功")

    records, summary, keys = driver.execute_query(
        "RETURN 1 AS test",
        database_="neo4j"
    )

    print("查询结果 =", records[0]["test"])

except Exception as e:
    print("❌ Neo4j Aura 连接失败")
    print("错误类型 =", type(e).__name__)
    print("错误信息 =", str(e))

finally:
    driver.close()