import copy
import time
import numpy as np
import os
import pickle
from sklearn.cluster import DBSCAN  # Import DBSCAN
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

class DPFLA:
    def _init_(self):
        pass

    def score(self, global_model, local_models, clients_types, selected_clients, p, w):

        n = len(selected_clients)
        P = generate_orthogonal_matrix(n=p, reuse=True)
        W = generate_orthogonal_matrix(n=n * n, reuse=True)
        Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(n)]

        param_diff = []
        param_diff_mask = []

        m_len = len(local_models)

        detect_res_list = []
        start_model_layer_param_list = []

        for idx in range(10):
            start_model_layer_param_list.append(list(global_model.state_dict()['fc2.weight'][idx].cpu()))
            # 计算每个本地模型的权重与全局模型最后一层权重之间的梯度差
            for i in range(m_len):
                end_model_layer_param = list(local_models[i].state_dict()['fc2.weight'][idx].cpu())
                gradient = calculate_parameter_gradients(start_model_layer_param_list[idx], end_model_layer_param)
                gradient = gradient.flatten()
                X_mask = Ws[i] @ gradient.reshape(1, -1) @ P
                param_diff_mask.append(X_mask)

            Z_mask = sum(param_diff_mask)
            U_mask, sigma, VT_mask = svd(Z_mask)

            G = Ws[0]
            for idx, val in enumerate(selected_clients):
                if idx == 0:
                    continue
                G = np.concatenate((G, Ws[idx]), axis=1)

            U = np.linalg.inv(G) @ U_mask
            U = U[:, :2]
            res = U * sigma[:2]
            detect_res_list.append(res)

        scores = batch_detect_outliers_dbscan(detect_res_list)

        logger.debug("-------------------------------------")
        logger.debug("Defense result:")

        final_scores = scores[-1]  # Extracting the final scores

        for i, pt in enumerate(clients_types):
            logger.info(f"{pt} scored {final_scores[i]}")


        bad_update_count = np.sum(np.logical_and(np.array(clients_types) == "Bad update", final_scores == 0))
        good_update_count = np.sum(np.logical_and(np.array(clients_types) == "Good update", final_scores == 1))
        total_errored = bad_update_count + good_update_count
        
        total_updates =len(clients_types)
        percentage_bad_updates_scored_one = (total_errored / total_updates) * 100 if total_updates > 0 else 0
        
        true_labels = np.array([1 if ct == "Good update" else 0 for ct in clients_types])
        predicted_labels = np.array(final_scores)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        
        logger.debug("Number of Bad updates scoring 0: ".format(bad_update_count))
        logger.debug("Number of Good updates scoring 1: ".format(good_update_count))

        logger.debug("Clustering Accuracy: {:.2f}%".format(percentage_bad_updates_scored_one))
        logger.debug("Clustering Precision: {:.4f}".format(precision))
        logger.debug("Clustering Recall: {:.4f}".format(recall))
        logger.debug("Clustering F1-score: {:.4f}".format(f1))

        # 返回得分列表
        return final_scores ,np.where(scores == 1)[0]


def generate_orthogonal_matrix(n, reuse=False, block_size=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_size is not None:
        file_name += '_blc%s' % block_size

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_size is not None:
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_size=sub_n)
                index = int(np.sum(qs[:i]))
                q[index:index + sub_n, index:index + sub_n] += tmp
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def calculate_parameter_gradients(params_1, params_2):
    return np.array([x for x in np.subtract(params_1, params_2)])


def batch_detect_outliers_dbscan(list, eps=0.5, min_samples=5):
    scores_list = []

    for data in list:
        # 训练DBSCAN模型
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        labels = dbscan.labels_
        scores = np.where(labels == -1, 0, 1)  # Assign 0 for outliers, 1 for inliers
        scores_list.append(scores)
    return scores_list


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def draw(data, clients_types, scores):
    SAVE_NAME = str(time.time()) + '.jpg'

    fig = plt.figure(figsize=(20, 6))
    fig1 = plt.subplot(121)
    for i, pt in enumerate(clients_types):
        if pt == 'Good update':
            plt.scatter(data[i, 0], data[i, 1], facecolors='none', edgecolors='black', marker='o', s=800,
                        label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], facecolors='black', edgecolors='black', marker='o', s=800,
                        label="Bad update")

    fig2 = plt.subplot(122)
    for i, pt in enumerate(clients_types):
        if scores[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], color="orange", s=800, label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", marker="x", linewidth=3, s=800, label="Bad update")

    plt.grid(False)
    # plt.show()
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1, dpi=400)