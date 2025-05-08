import os

os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import networkx as nx
import dgl
from dgl.nn import GraphConv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


class Neo4jGraphFraudDetector:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print("Connected to Neo4j successfully!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def extract_graph_features(self):
        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:HAS_CC]->(card:Card)
        OPTIONAL MATCH (u)-[:HAS_IP]->(ip:IP)
        OPTIONAL MATCH (u)-[:P2P]->(other:User)
        OPTIONAL MATCH (u)-[:REFERRED]->(referred:User)

        WITH u, 
             COUNT(DISTINCT card) AS card_count,
             COUNT(DISTINCT ip) AS ip_count,
             COUNT(DISTINCT other) AS p2p_connections,
             COUNT(DISTINCT referred) AS referral_count,
             u.is_flagged AS is_flagged,
             u.fraud_risk AS fraud_risk

        RETURN 
            id(u) AS user_id,
            card_count,
            ip_count,
            p2p_connections,
            referral_count,
            is_flagged,
            fraud_risk
        """
        with self.driver.session() as session:
            result = list(session.run(query))
        df = pd.DataFrame([dict(record) for record in result])

        feature_columns = ['card_count', 'ip_count', 'p2p_connections', 'referral_count']
        for col in feature_columns + ['is_flagged', 'fraud_risk']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        features = df[feature_columns].values.astype(np.float32)
        labels = df['fraud_risk'].where(df['fraud_risk'] != 0, df['is_flagged']).values.astype(np.int64)
        labels = (labels > 0).astype(np.int64)

        return features, labels, df['user_id'].values


class GraphFraudDetectionModel(nn.Module):
    def __init__(self, input_features, hidden_dim=64, output_dim=2):
        super(GraphFraudDetectionModel, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, features):
        x = self.feature_layers(features)
        return self.classifier(x)


def train_fraud_detection_model(features, labels):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GraphFraudDetectionModel(input_features=features.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    losses = []  # Danh sách lưu loss theo epoch
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)  # Chỉ lưu loss của mỗi epoch

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {total_loss / len(dataloader):.4f}, '
              f'Accuracy: {100 * correct_predictions / total_samples:.2f}%')

    return model, scaler, losses


def predict_fraud(model, scaler, features, user_ids):
    model.eval()
    scaled_features = scaler.transform(features)
    X_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)

    prediction_results = pd.DataFrame({
        'user_id': user_ids,
        'fraud_prediction': predictions.numpy()
    })

    print("Fraud Prediction Results:")
    print(prediction_results.head())
    return prediction_results

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.xticks(range(1, len(losses) + 1))  # Đảm bảo trục x đúng số epoch
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"

    try:
        detector = Neo4jGraphFraudDetector(uri, user, password)
        features, labels, user_ids = detector.extract_graph_features()

        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
        print("Unique labels:", np.unique(labels))

        model, scaler, losses = train_fraud_detection_model(features, labels)

        predictions = predict_fraud(model, scaler, features, user_ids)
        predictions.to_csv("fraud_predictions.csv", index=False)

        detector.close()
        print("Fraud detection model training and prediction completed!")
        # Gọi hàm vẽ sau khi huấn luyện
        plot_loss_curve(losses)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
