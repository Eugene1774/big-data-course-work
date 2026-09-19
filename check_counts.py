from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 检查所有标签的数量
    labels = ['Concept', 'Project', 'KnowledgeCard', 'Metric', 'Tag', 'Rule', 'Case', 'Method', 'Task', 'Hyperedge', 'Node', 'Card']
    for label in labels:
        result = s.run(f'MATCH (n:{label}) RETURN count(n) as cnt').single()
        print(f'{label}: {result["cnt"]}')
driver.close()
