from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')
database = ""

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session(database=database) as s:
    print("Dropping concept_name_unique constraint...")
    try:
        s.run("DROP CONSTRAINT concept_name_unique")
        print("Done!")
    except Exception as e:
        print(f"Error (may not exist): {e}")

driver.close()
