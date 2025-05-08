import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import networkx as nx
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Thiết lập backend DGL
os.environ["DGLBACKEND"] = "pytorch"


# Kết nối đến Neo4j
class Neo4jConnector:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print("Connected to Neo4j successfully!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")

    def close(self):
        self.driver.close()

    def run_query(self, query):
        with self.driver.session() as session:
            return list(session.run(query))


connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "12345678")

# Truy vấn dữ liệu từ Neo4j
node_query = """
MATCH (n) WHERE n:User OR n:FlaggedUser OR n:FraudRiskUser 
RETURN id(n) AS node_id, labels(n) AS node_labels, 
       n.age AS age, n.transactionCount AS transaction_count,
       n.accountAge AS account_age, n.loginFrequency AS login_frequency
"""
nodes = list(connector.run_query(node_query))

node_ids, node_features, labels = [], [], []
for record in nodes:
    node_id = record["node_id"]
    node_ids.append(node_id)

    features = torch.tensor([
        record.get("age", 30) if record.get("age") is not None else 30,
        record.get("transaction_count", 0) if record.get("transaction_count") is not None else 0,
        record.get("account_age", 0) if record.get("account_age") is not None else 0,
        record.get("login_frequency", 0) if record.get("login_frequency") is not None else 0
    ], dtype=torch.float32)
    node_features.append(features)

    labels.append(1 if "FlaggedUser" in record["node_labels"] or "FraudRiskUser" in record["node_labels"] else 0)

# Chuyển đổi danh sách sang Tensor
torch_feature_matrix = torch.stack(node_features)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Truy vấn cạnh từ Neo4j
edge_query = """
MATCH (n1)-[r]->(n2) WHERE n1:User OR n1:FlaggedUser OR n1:FraudRiskUser 
RETURN id(n1) AS source, id(n2) AS target
"""
edges = list(connector.run_query(edge_query))
edge_list = [(record["source"], record["target"]) for record in edges]

# Xây dựng đồ thị bằng NetworkX
graph = nx.DiGraph()
node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
graph.add_nodes_from(range(len(node_ids)))
for src, tgt in edge_list:
    if src in node_id_to_idx and tgt in node_id_to_idx:
        graph.add_edge(node_id_to_idx[src], node_id_to_idx[tgt])

dgl_graph = dgl.from_networkx(graph)
dgl_graph = dgl.add_self_loop(dgl_graph)
dgl_graph.ndata['features'] = torch_feature_matrix
dgl_graph.ndata['labels'] = labels_tensor

# Chia tập huấn luyện và kiểm thử
indices = np.arange(len(labels))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
train_mask, test_mask = torch.zeros(len(labels), dtype=torch.bool), torch.zeros(len(labels), dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True
dgl_graph.ndata['train_mask'], dgl_graph.ndata['test_mask'] = train_mask, test_mask


# Định nghĩa mô hình GNN
class FraudDetectionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, g, features):
        x = self.activation(self.norm1(self.conv1(g, features)))
        x = self.dropout(x)
        x = self.activation(self.norm2(self.conv2(g, x)))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x


model = FraudDetectionGNN(torch_feature_matrix.shape[1], 64, 2)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Dự đoán và đếm số lượng nút gian lận
model.eval()
with torch.no_grad():
    logits = model(dgl_graph, dgl_graph.ndata['features'])
    predictions = torch.argmax(logits, dim=1)
    fraud_count = (predictions == 1).sum().item()
    print(f"Số lượng nút gian lận được phát hiện: {fraud_count}")
    print(f"Số lượng nút thực sự là gian lận: {(labels_tensor == 1).sum().item()}")

def plot_loss_curve(losses):
    """
    Vẽ biểu đồ Cross Entropy Loss theo số epoch.
    """
    df = pd.DataFrame({'Epoch': range(1, len(losses) + 1), 'Loss': losses})

    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Loss'], marker='o', linestyle='-', color='b', label='Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
# Huấn luyện mô hình
num_epochs = 20
train_losses = []
test_accs = []
epochs = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(dgl_graph, dgl_graph.ndata['features'])
    loss = criterion(logits[train_mask], labels_tensor[train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(loss.item())
    epochs.append(epoch + 1)

    with torch.no_grad():
        model.eval()
        test_logits = model(dgl_graph, dgl_graph.ndata['features'])
        test_pred = torch.argmax(test_logits[test_mask], dim=1)
        test_acc = accuracy_score(labels_tensor[test_mask].cpu(), test_pred.cpu())

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Test Acc: {test_acc:.4f}")

# Sau khi huấn luyện xong
plot_loss_curve(train_losses)




