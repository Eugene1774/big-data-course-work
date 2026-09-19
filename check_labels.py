from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    result = s.run('CALL db.labels() YIELD label RETURN collect(label) as labels').single()
    print(f'数据库中的标签: {result["labels"]}')
driver.close()
