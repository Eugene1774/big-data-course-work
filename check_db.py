from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=3600)
with driver.session() as s:
    result = s.run('MATCH (n) RETURN count(n) as total').single()
    print(f'总节点数: {result["total"]}')
    
    result = s.run('MATCH (p:Project) RETURN count(p) as cnt').single()
    print(f'Project数: {result["cnt"]}')
    
    result = s.run('MATCH (c:Concept) RETURN count(c) as cnt').single()
    print(f'Concept数: {result["cnt"]}')
    
    result = s.run('MATCH (k:KnowledgeCard) RETURN count(k) as cnt').single()
    print(f'KnowledgeCard数: {result["cnt"]}')
    
    result = s.run('MATCH ()-[r]->() RETURN count(r) as cnt').single()
    print(f'总关系数: {result["cnt"]}')
driver.close()
