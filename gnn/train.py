import os

import neo4j

os.environ["DGLBACKEND"] = "pytorch"
from neo4j import GraphDatabase
import dgl
import torch as th
import numpy as np
import time
import copy

from sklearn.metrics import confusion_matrix
from estimator_fns import *
from graph_utils import *
from data import *
from utils import *
from pytorch_model import *

from datetime import date

def normalize_data(data):
    nodes = {}
    edges = []

    for record in data:
        # Chuẩn hóa nút
        node_id = record['node_id']
        if node_id not in nodes:
            properties = record['properties']
            # Chuyển đổi neo4j.time.Date (nếu cần)
            for key, value in properties.items():
                if isinstance(value, neo4j.time.Date):
                    properties[key] = value.to_native()
            nodes[node_id] = {
                'node_id': node_id,
                'labels': record['labels'],
                'properties': properties
            }

        # Chuẩn hóa cạnh
        dst = record.get('dst')
        edge_type = record.get('edge_type')
        if dst is not None and edge_type is not None:
            edges.append({'src': node_id, 'dst': dst, 'edge_type': edge_type})

    return nodes, edges

def fetch_data_from_neo4j(uri, user, password, query):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)
    # print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1

def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = th.argmax(preds, axis=1).numpy()
    precision, recall, f1 = get_f1_score(labels, preds)

    return f1

def normalize(feature_matrix):
    mean = th.mean(feature_matrix, axis=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev


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

    # Xử lý kết quả: Tách nút và cạnh
    nodes = {}
    edges = []

    for record in data:
        node_id = record['node_id']
        if node_id not in nodes:
            nodes[node_id] = {
                'node_id': node_id,
                'labels': record['labels'],
                'properties': record['properties']
            }

        # Nếu có cạnh, thêm vào danh sách edges
        if record['dst'] is not None and record['edge_type'] is not None:
            edges.append({
                'src': node_id,
                'dst': record['dst'],
                'edge_type': record['edge_type']
            })

    # Đảm bảo tất cả các nút trong edges cũng có trong nodes
    for edge in edges:
        if edge['src'] not in nodes:
            nodes[edge['src']] = {
                'node_id': edge['src'],
                'labels': ['Unknown'],  # Gán nhãn mặc định
                'properties': {}
            }
        if edge['dst'] not in nodes:
            nodes[edge['dst']] = {
                'node_id': edge['dst'],
                'labels': ['Unknown'],  # Gán nhãn mặc định
                'properties': {}
            }

    # Create node mapping
    id_to_node = {node['node_id']: i for i, node in enumerate(nodes.values())}

    # Add features and node types
    node_features = []
    node_types = []
    max_feature_length = 0

    for node in nodes.values():
        node_types.append(node['labels'][0])  # Assuming single label
        feature_vector = np.array([v for v in node['properties'].values() if isinstance(v, (int, float))])
        node_features.append(feature_vector)
        max_feature_length = max(max_feature_length, len(feature_vector))

    # Pad all feature vectors to have the same length
    for i, feature_vector in enumerate(node_features):
        if len(feature_vector) < max_feature_length:
            padded_vector = np.pad(feature_vector, (0, max_feature_length - len(feature_vector)), 'constant')
            node_features[i] = padded_vector

    # Convert to tensor
    node_features = np.array(node_features)
    node_features = th.tensor(node_features, dtype=th.float32)

    # Normalize features
    mean, std, node_features = normalize(node_features)

    # Process edges
    edge_list = [(id_to_node[e['src']], id_to_node[e['dst']], e['edge_type']) for e in edges]

    # Construct DGL graph
    data_dict = {}
    for src, dst, etype in edge_list:
        if etype not in data_dict:
            data_dict[(src, etype, dst)] = []
        data_dict[(src, etype, dst)].append((src, dst))

    g = dgl.heterograph(data_dict)

    # Assign features
    g.ndata['features'] = node_features
    return g, mean, std


def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):
    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)
    return model

def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask, device, n_epochs, thresh,
             compute_metrics=True):
    """
    A full graph version of RGCN training
    """
    duration = []
    best_loss = 1
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))

        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f} ".format(
            epoch, np.mean(duration), loss_val, metric))

        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(epoch, np.mean(duration), loss_val, metric)
        with open('./output/results.txt', 'a+') as f:
            f.write(epoch_result)

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)

    class_preds, pred_proba = get_model_class_predictions(best_model, test_g, features, labels, device,
                                                          threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(class_preds, pred_proba, labels.numpy(), test_mask.numpy(),
                                                     './output/')
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba

def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = th.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).detach().numpy(), pred_proba[:,1].detach().numpy()
    return np.where(pred_proba.detach().numpy() > threshold, 1, 0), pred_proba[:,1].detach().numpy()



if __name__ == '__main__':
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"

    # Load data
    g, mean, std = construct_graph_from_neo4j(uri, user, password)

    print(f"Graph loaded with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

    # Sử dụng CPU
    device = th.device('cpu')
    print(f"Using device: {device}")

    # Prepare model and data
    in_feats = g.ndata['features'].shape[1]
    n_classes = 2  # Example binary classification
    ntype_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}

    model = get_model(ntype_dict, g.etypes, {'n_hidden': 16, 'n_layers': 3}, in_feats, n_classes, device)
    optim = th.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss = th.nn.CrossEntropyLoss()

    features = g.ndata['features'].to(device)
    labels = th.randint(0, n_classes, (g.number_of_nodes(),)).to(device)  # Mock labels

    print("Starting Model training on CPU")
    model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g, None, device, n_epochs=10, thresh=0.5)

    print("Finished Model training")

    # Save model
    if not os.path.exists('./output'):
        os.makedirs('./output')

    print("Model and metadata saved.")


