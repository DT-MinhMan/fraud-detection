import os
os.environ["DGLBACKEND"] = "pytorch"
from neo4j import GraphDatabase

def fetch_data_from_neo4j(uri, user, password, query):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def construct_graph_from_neo4j(uri, user, password):
    # Gộp truy vấn nodes và edges
    query = """
    MATCH (n:User)
    OPTIONAL MATCH (n:User)-[r]->(m)
    RETURN 
        ID(n) AS node_id, 
        labels(n) AS labels, 
        properties(n) AS properties,
        ID(m) AS dst, 
        type(r) AS edge_type
    LIMIT 1000
    """
    # Fetch data từ Neo4j
    data = fetch_data_from_neo4j(uri, user, password, query)

    if not data:
        print("No data returned from query.")
    else:
        print(f"Data returned: {len(data)} records")
        print(data[:5])  # Chỉ in 5 bản ghi đầu tiên để kiểm tra

if __name__ == '__main__':
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    construct_graph_from_neo4j(uri, user, password)

