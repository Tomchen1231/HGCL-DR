import torch as th
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, roc_curve, roc_auc_score)

import numpy as np
import torch.nn.functional as fn
def evaluate(args, model, graph_data,
             drug_graph, drug_feat, drug_sim_feat,
             dis_graph, dis_feat, dis_sim_feat,subgraphs):
    # rating_values = dataset.test_truths
    rating_values = graph_data['test'][2]
    # enc_graph = dataset.test_enc_graph.int().to(args.device)
    # dec_graph = dataset.test_dec_graph.int().to(args.device)
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    with th.no_grad():
        test_score, _, _, _, _, out1 = model(enc_graph, dec_graph,
                                         drug_graph, drug_sim_feat, drug_feat,
                                         dis_graph, dis_sim_feat, dis_feat, subgraphs)
    y_score = out1.view(-1).cpu().tolist()
    test_prob = fn.softmax(out1, dim=-1)
    out1 = th.argmax(out1, dim=-1)
    test_prob = test_prob[:, 1]
    y_true = rating_values.cpu().tolist()
    test_prob = test_prob.cpu().numpy()
    out1 = out1.cpu().numpy()
    accuracy = accuracy_score(y_true, out1)
    mcc = matthews_corrcoef(y_true, out1)
    precision = precision_score(y_true, out1)
    recall = recall_score(y_true, out1)
    f1 = f1_score(y_true, out1)

    fpr, tpr, _ = roc_curve(y_true, test_prob )
    Auc = auc(fpr, tpr)

    precision1, recall1, _ = precision_recall_curve(y_true, test_prob )
    aupr = auc(recall1, precision1)
    # 使用数据分布中的阈值，例如使用中位数或平均值
    #threshold = np.median(y_score)  # 或 np.mean(y_score)
    #y_pred = [1 if score >= threshold else 0 for score in y_score]
    #precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    #aupr = metrics.auc(recall, precision)
    #f1 = metrics.f1_score(fpr, tpr)
    return Auc, aupr, f1, precision, recall, y_true, y_score
