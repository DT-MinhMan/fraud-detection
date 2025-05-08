import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from neo4j import GraphDatabase
import numpy as np
import torch.optim as optim
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class Neo4jConnector:
    """
    Lớp kết nối với cơ sở dữ liệu Neo4j
    """

    def __init__(self, uri, user, password):
        # Khởi tạo kết nối tới Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to Neo4j successfully!")

    def close(self):
        # Đóng kết nối
        self.driver.close()

    def run_query(self, query):
        # Thực thi truy vấn và trả về kết quả
        with self.driver.session() as session:
            return list(session.run(query))


class HeteroGNN(nn.Module):
    """
    Mô hình Graph Neural Network cho đồ thị dị loại (có nhiều loại nút và cạnh khác nhau)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(HeteroGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # Tạo các lớp tích chập đồ thị cho từng loại mối quan hệ
            conv = dglnn.HeteroGraphConv({
                'HAS_CC': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Người dùng có thẻ tín dụng
                'HAS_IP': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Người dùng có địa chỉ IP
                'P2P': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Quan hệ người dùng với người dùng
                'P2P_WITH_SHARED_CARD': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Người dùng chung thẻ
                'REFERRED': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Người dùng được giới thiệu bởi
                'SHARED_IDS': dglnn.SAGEConv(input_size, hidden_size, 'mean'),  # Người dùng dùng chung ID
                'USED': dglnn.SAGEConv(input_size, hidden_size, 'mean')  # Người dùng sử dụng thiết bị
            }, aggregate='mean')
            self.convs.append(conv)
            # Chuẩn hóa lớp để ổn định quá trình huấn luyện
            self.norms.append(nn.LayerNorm(hidden_size))
            input_size = hidden_size

        self.dropout = nn.Dropout(dropout)

        # Bộ phân loại MLP để dự đoán người dùng có gian lận hay không
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, g, features):
        """
        Truyền dữ liệu qua mạng neural
        """
        h = features
        for conv, norm in zip(self.convs, self.norms):
            h = conv(g, h)  # Truyền qua lớp tích chập đồ thị
            h = {ntype: F.relu(norm(feat)) for ntype, feat in h.items()}  # Áp dụng ReLU và chuẩn hóa lớp
            h = {ntype: self.dropout(feat) for ntype, feat in h.items()}  # Áp dụng dropout

        h_user = h['User']  # Lấy đặc trưng của nút người dùng
        return self.classifier(h_user)  # Phân loại người dùng


def preprocess_data(results):
    """
    Tiền xử lý dữ liệu từ kết quả truy vấn Neo4j
    """
    node_features = {}  # Lưu đặc trưng của các nút
    node_labels = {}  # Lưu nhãn của các nút
    edge_list = []  # Lưu danh sách cạnh

    for record in results:
        # Trích xuất thông tin nút từ kết quả truy vấn
        for node_id, labels_list, feat1, feat2 in [
            (record["node1_id"], record["node1_labels"], record["node1_feature1"], record["node1_feature2"]),
            (record["node2_id"], record["node2_labels"], record["node2_feature1"], record["node2_feature2"])
        ]:
            if node_id not in node_features:
                # Lưu đặc trưng của nút (thay thế None bằng 0)
                node_features[node_id] = {
                    "feature1": feat1 if feat1 is not None else 0,
                    "feature2": feat2 if feat2 is not None else 0
                }
                node_labels[node_id] = labels_list

        # Thêm thông tin cạnh vào danh sách
        edge_list.append((record["source"], record["target"], record["relation_type"]))

    return node_features, node_labels, edge_list


def create_heterograph(node_features, node_labels, edge_list):
    """
    Tạo đồ thị dị loại từ dữ liệu đã tiền xử lý
    """
    # Phân loại nút theo loại
    user_nodes = [nid for nid in node_features if "User" in node_labels[nid]]
    card_nodes = [nid for nid in node_features if "Card" in node_labels[nid]]
    ip_nodes = [nid for nid in node_features if "IP" in node_labels[nid]]
    device_nodes = [nid for nid in node_features if "Device" in node_labels[nid]]

    # Ánh xạ ID nút gốc sang chỉ số mới
    user_mapping = {old_id: new_idx for new_idx, old_id in enumerate(user_nodes)}
    card_mapping = {old_id: new_idx for new_idx, old_id in enumerate(card_nodes)}
    ip_mapping = {old_id: new_idx for new_idx, old_id in enumerate(ip_nodes)}
    device_mapping = {old_id: new_idx for new_idx, old_id in enumerate(device_nodes)}

    # Chuẩn bị dữ liệu cạnh cho đồ thị dị loại
    hetero_graph_data = {
        ('User', 'HAS_CC', 'Card'): [],
        ('User', 'HAS_IP', 'IP'): [],
        ('User', 'P2P', 'User'): [],
        ('User', 'P2P_WITH_SHARED_CARD', 'User'): [],
        ('User', 'REFERRED', 'User'): [],
        ('User', 'SHARED_IDS', 'User'): [],
        ('User', 'USED', 'Device'): []
    }

    # Định nghĩa ánh xạ từ loại quan hệ đến loại nút đích
    relation_to_target = {
        'HAS_CC': ('Card', card_mapping),
        'HAS_IP': ('IP', ip_mapping),
        'P2P': ('User', user_mapping),
        'P2P_WITH_SHARED_CARD': ('User', user_mapping),
        'REFERRED': ('User', user_mapping),
        'SHARED_IDS': ('User', user_mapping),
        'USED': ('Device', device_mapping)
    }

    # Xử lý từng cạnh và thêm vào đồ thị
    for src, tgt, rel in edge_list:
        if src not in user_mapping or rel not in relation_to_target:
            continue

        target_type, tgt_mapping = relation_to_target[rel]
        if tgt not in tgt_mapping:
            continue

        # Thêm cạnh vào đồ thị dựa trên loại đích
        if target_type == 'User':
            hetero_graph_data[('User', rel, 'User')].append((user_mapping[src], user_mapping[tgt]))
        elif target_type == 'Card':
            hetero_graph_data[('User', rel, 'Card')].append((user_mapping[src], card_mapping[tgt]))
        elif target_type == 'IP':
            hetero_graph_data[('User', rel, 'IP')].append((user_mapping[src], ip_mapping[tgt]))
        elif target_type == 'Device':
            hetero_graph_data[('User', rel, 'Device')].append((user_mapping[src], device_mapping[tgt]))

    # Định nghĩa số lượng nút cho mỗi loại
    num_nodes_dict = {
        'User': len(user_nodes),
        'Card': len(card_nodes),
        'IP': len(ip_nodes),
        'Device': len(device_nodes)
    }

    # Tạo đồ thị dị loại
    hetero_graph = dgl.heterograph(hetero_graph_data, num_nodes_dict=num_nodes_dict)

    # Chuẩn bị đặc trưng nút
    def prepare_features(node_type, nodes):
        """Hàm nội bộ để chuẩn bị đặc trưng cho mỗi loại nút"""
        features = np.array([
            [node_features[old_id]["feature1"], node_features[old_id]["feature2"]]
            for old_id in nodes
        ])
        hetero_graph.nodes[node_type].data['features'] = torch.tensor(features, dtype=torch.float32)

    # Áp dụng đặc trưng cho từng loại nút
    prepare_features('User', user_nodes)
    prepare_features('Card', card_nodes)
    prepare_features('IP', ip_nodes)
    prepare_features('Device', device_nodes)

    # Chuẩn bị nhãn cho nút User (1 nếu là gian lận, 0 nếu không)
    user_labels = [1 if "FlaggedUser" in node_labels[old_id] else 0 for old_id in user_nodes]
    labels_tensor = torch.tensor(user_labels, dtype=torch.long)

    return hetero_graph, labels_tensor, user_nodes, user_mapping


def create_train_test_masks(num_nodes, labels, test_size=0.2, stratify=True, random_state=42):
    """
    Tạo mặt nạ huấn luyện/kiểm tra cho các nút.

    Args:
        num_nodes (int): Số lượng nút
        labels (torch.Tensor): Nhãn của nút
        test_size (float): Tỷ lệ nút dùng để kiểm tra
        stratify (bool): Có phân tầng theo nhãn lớp hay không
        random_state (int): Hạt giống ngẫu nhiên

    Returns:
        Tuple gồm mặt nạ huấn luyện và kiểm tra dưới dạng tensor logic
    """
    indices = np.arange(num_nodes)
    if stratify:
        # Phân tầng giúp giữ tỷ lệ các lớp trong tập train và test
        stratify_data = labels.cpu().numpy()
    else:
        stratify_data = None

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_data,
        random_state=random_state
    )

    # Tạo mặt nạ dưới dạng tensor logic
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask


def evaluate_model(model, graph, features, labels, mask):
    """
    Đánh giá hiệu suất mô hình
    """
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]

        # Lấy xác suất của lớp gian lận (lớp 1)
        probs = F.softmax(logits, dim=1)[:, 1]
        # Hạ ngưỡng xuống để tăng dự đoán lớp gian lận (mặc định là 0.5)
        threshold = 0.3
        predicted = (probs >= threshold).long()

        # Tính các chỉ số đánh giá
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu().numpy(),
            predicted.cpu().numpy(),
            average='binary',
            zero_division=1  # Tránh lỗi chia cho 0
        )

        # Tính diện tích dưới đường cong ROC
        auc = roc_auc_score(labels.cpu().numpy(), logits[:, 1].cpu().numpy())

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    # In số lượng dự đoán dương (gian lận)
    num_positives = np.sum(predicted.cpu().numpy() == 1)
    print(f"Number of positive predictions: {num_positives}")

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

def train_model(model, graph, features, labels, train_mask, test_mask, device, epochs=100, lr=1e-3, weight_decay=1e-5):
    """
    Huấn luyện mô hình GNN
    """
    # Khởi tạo bộ tối ưu hóa và lịch giảm learning rate
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    losses = []  # Danh sách lưu loss theo epoch

    # Theo dõi hiệu suất tốt nhất và cài đặt early stopping
    best_val_f1 = 0
    patience = 10
    counter = 0
    best_model_state = None

    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Dự đoán và tính loss
        predictions = model(graph, features)
        loss = criterion(predictions[train_mask], labels[train_mask])

        # Lan truyền ngược và cập nhật tham số
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Lưu giá trị loss của epoch hiện tại
        losses.append(loss.item())

        # Đánh giá hiệu suất trên tập huấn luyện và kiểm tra
        train_metrics = evaluate_model(model, graph, features, labels, train_mask)
        test_metrics = evaluate_model(model, graph, features, labels, test_mask)

        # In thông tin hiệu suất
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {loss.item():.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}")

        # Lưu mô hình tốt nhất và kiểm tra early stopping
        if test_metrics['f1'] > best_val_f1:
            best_val_f1 = test_metrics['f1']
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Tải lại mô hình tốt nhất
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with F1: {best_val_f1:.4f}")

    return model, losses


def main():
    """
    Hàm chính để chạy toàn bộ quá trình
    """
    # Đặt hạt giống ngẫu nhiên để kết quả có thể tái tạo
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Thông tin kết nối Neo4j
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"

    # Kết nối và truy vấn dữ liệu
    connector = Neo4jConnector(uri, user, password)

    # Truy vấn để lấy thông tin về nút và cạnh từ Neo4j
    combined_query = """
    MATCH (n1)-[r]->(n2)
    RETURN id(n1) AS source, id(n2) AS target, type(r) AS relation_type,
           id(n1) AS node1_id, labels(n1) AS node1_labels, n1.property1 AS node1_feature1, n1.property2 AS node1_feature2,
           id(n2) AS node2_id, labels(n2) AS node2_labels, n2.property1 AS node2_feature1, n2.property2 AS node2_feature2
    """
    results = connector.run_query(combined_query)
    connector.close()

    # Tiền xử lý dữ liệu
    node_features, node_labels, edge_list = preprocess_data(results)
    hetero_graph, labels_tensor, user_nodes, user_mapping = create_heterograph(node_features, node_labels, edge_list)

    # Tạo mặt nạ huấn luyện và kiểm tra
    num_user_nodes = hetero_graph.num_nodes('User')
    train_mask, test_mask = create_train_test_masks(num_user_nodes, labels_tensor, test_size=0.2)

    # Chuyển dữ liệu sang GPU nếu có
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = {ntype: hetero_graph.nodes[ntype].data['features'].to(device) for ntype in hetero_graph.ntypes}
    hetero_graph = hetero_graph.to(device)
    labels_tensor = labels_tensor.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    # Khởi tạo và huấn luyện mô hình
    model = HeteroGNN(input_size=features['User'].shape[1], hidden_size=128, output_size=2).to(device)
    trained_model, losses = train_model(model, hetero_graph, features, labels_tensor, train_mask, test_mask, device)

    # Đếm số lượng người dùng gian lận trong dữ liệu kiểm tra
    test_user_indices = torch.where(test_mask)[0]
    test_labels = labels_tensor[test_user_indices]
    fraud_count = torch.sum(test_labels == 1).item()
    total_test_users = len(test_user_indices)

    # In thống kê về gian lận trong dữ liệu kiểm tra
    print(f"\nFraud Statistics in Test Data:")
    print(f"Total test users: {total_test_users}")
    print(f"Fraudulent test users: {fraud_count} ({fraud_count / total_test_users:.2%})")
    print(
        f"Legitimate test users: {total_test_users - fraud_count} ({(total_test_users - fraud_count) / total_test_users:.2%})")

    # In ID của người dùng gian lận
    fraud_test_indices = test_user_indices[test_labels == 1]
    fraud_original_ids = [user_nodes[idx] for idx in fraud_test_indices.cpu().numpy()]
    print(f"\nOriginal IDs of fraudulent users in test set:")
    print(fraud_original_ids)

    # In hiệu suất mô hình đặc biệt đối với các trường hợp gian lận
    with torch.no_grad():
        logits = model(hetero_graph, features)
        test_preds = torch.argmax(logits[test_mask], dim=1)
        test_labels = labels_tensor[test_mask]

        # Lấy chỉ số dành riêng cho trường hợp gian lận
        fraud_indices = (test_labels == 1)
        fraud_correct = torch.sum((test_preds == test_labels) & fraud_indices).item()
        fraud_total = torch.sum(fraud_indices).item()

        print(f"\nModel Performance on Fraud Cases:")
        print(f"Fraud detection accuracy: {fraud_correct / fraud_total:.4f} ({fraud_correct}/{fraud_total})")

    # Gọi hàm vẽ sau khi huấn luyện
    plot_loss_curve(losses)

if __name__ == "__main__":
    main()