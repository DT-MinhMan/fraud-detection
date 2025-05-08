import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split

###############################
# 1. Build the Heterograph    #
###############################

# Lớp kết nối đến cơ sở dữ liệu Neo4j
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to Neo4j successfully!")

    def close(self):
        self.driver.close()

    def run_query(self, query):
        with self.driver.session() as session:
            return list(session.run(query))

# Kết nối đến cơ sở dữ liệu Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
connector = Neo4jConnector(uri, user, password)

# Truy vấn dữ liệu từ Neo4j
combined_query = """
MATCH (n1)-[r]->(n2)
RETURN id(n1) AS source, id(n2) AS target, type(r) AS relation_type,
       id(n1) AS node1_id, labels(n1) AS node1_labels, n1.property1 AS node1_feature1, n1.property2 AS node1_feature2,
       id(n2) AS node2_id, labels(n2) AS node2_labels, n2.property1 AS node2_feature1, n2.property2 AS node2_feature2
"""
results = connector.run_query(combined_query)
connector.close()

# Xử lý dữ liệu
node_features, node_labels, edge_list = {}, {}, []
for record in results:
    for node_id, labels_list, feat1, feat2 in [
        (record["node1_id"], record["node1_labels"], record["node1_feature1"], record["node1_feature2"]),
        (record["node2_id"], record["node2_labels"], record["node2_feature1"], record["node2_feature2"])
    ]:
        if node_id not in node_features:
            node_features[node_id] = {"feature1": feat1 or 0, "feature2": feat2 or 0}
            node_labels[node_id] = labels_list
    edge_list.append((record["source"], record["target"], record["relation_type"]))

# Phân loại node
def get_nodes_by_label(label):
    return [nid for nid in node_features if label in node_labels[nid]]

user_nodes, card_nodes = get_nodes_by_label("User"), get_nodes_by_label("Card")
ip_nodes, device_nodes = get_nodes_by_label("IP"), get_nodes_by_label("Device")

# Tạo mapping node
user_mapping = {old_id: new_idx for new_idx, old_id in enumerate(user_nodes)}
card_mapping = {old_id: new_idx for new_idx, old_id in enumerate(card_nodes)}
ip_mapping = {old_id: new_idx for new_idx, old_id in enumerate(ip_nodes)}
device_mapping = {old_id: new_idx for new_idx, old_id in enumerate(device_nodes)}

# Xây dựng heterograph
def map_edges(edge_list, mapping, relation_type, target_mapping):
    return [(mapping[src], target_mapping[tgt]) for src, tgt, rel in edge_list if src in mapping and tgt in target_mapping and rel == relation_type]

hetero_graph_data = {
    ('User', 'HAS_CC', 'Card'): map_edges(edge_list, user_mapping, 'HAS_CC', card_mapping),
    ('User', 'HAS_IP', 'IP'): map_edges(edge_list, user_mapping, 'HAS_IP', ip_mapping),
    ('User', 'P2P', 'User'): map_edges(edge_list, user_mapping, 'P2P', user_mapping),
    ('User', 'USED', 'Device'): map_edges(edge_list, user_mapping, 'USED', device_mapping)
}
hetero_graph = dgl.heterograph(hetero_graph_data, num_nodes_dict={'User': len(user_nodes), 'Card': len(card_nodes), 'IP': len(ip_nodes), 'Device': len(device_nodes)})

# Gán đặc trưng node
for node_type, node_list in zip(['User', 'Card', 'IP', 'Device'], [user_nodes, card_nodes, ip_nodes, device_nodes]):
    features = np.array([[node_features[old_id]["feature1"], node_features[old_id]["feature2"]] for old_id in node_list])
    hetero_graph.nodes[node_type].data['features'] = torch.tensor(features, dtype=torch.float32)

# Gán nhãn user
labels_tensor = torch.tensor([1 if "FlaggedUser" in node_labels[nid] else 0 for nid in user_nodes], dtype=torch.long)

#####################################
# 2. Train-Test Split and Training #
#####################################
train_idx, test_idx, train_labels, test_labels = train_test_split(
    np.arange(len(user_nodes)), labels_tensor.numpy(), test_size=0.2, random_state=42, stratify=labels_tensor.numpy()
)
train_labels_tensor, test_labels_tensor = torch.tensor(train_labels, dtype=torch.long), torch.tensor(test_labels, dtype=torch.long)

class HeteroGNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HeteroGNN, self).__init__()
        # Định nghĩa GraphConv cho tất cả các quan hệ trong đồ thị dị thể
        self.conv1 = dglnn.HeteroGraphConv({
            'P2P': dglnn.GraphConv(input_size, hidden_size),
            'HAS_CC': dglnn.GraphConv(input_size, hidden_size),
            'HAS_IP': dglnn.GraphConv(input_size, hidden_size),
            'USED': dglnn.GraphConv(input_size, hidden_size)
        }, aggregate='mean')  # Dùng phép tổng hợp trung bình

        self.conv2 = dglnn.HeteroGraphConv({
            'P2P': dglnn.GraphConv(hidden_size, hidden_size),
            'HAS_CC': dglnn.GraphConv(hidden_size, hidden_size),
            'HAS_IP': dglnn.GraphConv(hidden_size, hidden_size),
            'USED': dglnn.GraphConv(hidden_size, hidden_size)
        }, aggregate='mean')

        self.fc = nn.Linear(hidden_size, output_size)  # Lớp fully-connected

    def forward(self, g, features):
        h = self.conv1(g, features)  # Truyền dữ liệu qua lớp HeteroGraphConv
        h = {ntype: F.relu(feat) for ntype, feat in h.items()}  # Áp dụng ReLU
        h = self.conv2(g, h)  # Thêm một tầng GNN nữa
        h = {ntype: F.relu(feat) for ntype, feat in h.items()}
        return self.fc(h['User'])  # Lấy đặc trưng của 'User' để dự đoán


# Khởi tạo mô hình
input_size = hetero_graph.nodes['User'].data['features'].shape[1]
model = HeteroGNN(input_size, hidden_size=64, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Tính toán trọng số cho loss function để tránh mất cân bằng dữ liệu
num_fraud = (labels_tensor == 1).sum().item()
num_non_fraud = (labels_tensor == 0).sum().item()
class_weights = torch.tensor([num_non_fraud / (num_fraud + num_non_fraud), num_fraud / (num_fraud + num_non_fraud)], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Huấn luyện mô hình
features = {ntype: hetero_graph.nodes[ntype].data['features'] for ntype in hetero_graph.ntypes}
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    predictions = model(hetero_graph, features)[train_idx]
    loss = criterion(predictions, train_labels_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Đánh giá mô hình
model.eval()
with torch.no_grad():
    # test_predictions = model(hetero_graph, features)[test_idx]
    # predicted_test_labels = torch.argmax(test_predictions, dim=1)
    # accuracy = (predicted_test_labels == test_labels_tensor).float().mean().item()
    # num_fraud = (predicted_test_labels == 1).sum().item()
    # num_non_fraud = (predicted_test_labels == 0).sum().item()
    # print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"Total Fraud Users Predicted: {num_fraud}")
    # print(f"Total Non-Fraud Users Predicted: {num_non_fraud}")

    all_predictions = model(hetero_graph, features)  # Dự đoán trên toàn bộ dataset
    predicted_labels = torch.argmax(all_predictions, dim=1)
    accuracy = (predicted_labels == labels_tensor).float().mean().item()
    num_fraud = (predicted_labels == 1).sum().item()
    num_non_fraud = (predicted_labels == 0).sum().item()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total Fraud Users Predicted (Entire Dataset): {num_fraud}")
    print(f"Total Non-Fraud Users Predicted (Entire Dataset): {num_non_fraud}")

    # num_fraud_labels = (labels_tensor == 1).sum().item()
    # num_non_fraud_labels = (labels_tensor == 0).sum().item()
    # print(f"Total flagged fraud nodes: {num_fraud_labels}")
    # print(f"Total non-fraud nodes: {num_non_fraud_labels}")

