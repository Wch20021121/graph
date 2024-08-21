import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def similarity_calculate(x: list):
    # 存储所有2个实体之间的相似度,矩阵形式
    sim_matrix = np.ones((len(x), len(x)))
    for xi in tqdm(range(len(x))):
        for xj in range(xi + 1, len(x)):
            # 计算相似度
            similarity = cosine_similarity([x[xi]], [x[xj]])[0][0]
            sim_matrix[xi][xj] = similarity
            sim_matrix[xj][xi] = similarity
    # 保存相似度矩阵
    return sim_matrix


def get_entity_similarity_matrix(data):
    entity_list = []
    for i in tqdm(range(len(data))):
        data[i][-1] = data[i][-1].replace('[', '').replace(']', '').replace('\n', '').split(' ')
        for j in range(len(data[i][-1]) - 1, -1, -1):
            if data[i][-1][j] == '':
                data[i][-1].pop(j)
            else:
                data[i][-1][j] = float(data[i][-1][j])
        data[i][-1] = np.array(data[i][-1])
        entity_list.append(data[i][-1])
    esm = similarity_calculate(entity_list)
    # 保存相似度矩阵
    np.save('data/matrix/entity_similarity_matrix.npy', esm)
    return esm


def get_event_similarity_matrix(data):
    event_list = []
    for i in tqdm(range(len(data))):
        data[i][-2] = data[i][-2].replace('[', '').replace(']', '').replace('\n', '').split(' ')
        data[i][-3] = data[i][-3].replace('[', '').replace(']', '').replace('\n', '').split(' ')
        for j in range(len(data[i][-2]) - 1, -1, -1):
            if data[i][-2][j] == '':
                data[i][-2].pop(j)
            else:
                data[i][-2][j] = float(data[i][-2][j])
        for j in range(len(data[i][-3]) - 1, -1, -1):
            if data[i][-3][j] == '':
                data[i][-3].pop(j)
            else:
                data[i][-3][j] = float(data[i][-3][j])
        data[i][-2] = np.array(data[i][-2])
        data[i][-3] = np.array(data[i][-3])
        event_list.append(data[i][-2])
        event_list.append(data[i][-3])
    esm = similarity_calculate(event_list)
    # 保存相似度矩阵
    np.save('data/matrix/event_similarity_matrix.npy', esm)
    return esm


def merge_entity(using_data, lower_limit: float = 0.8, upper_limit: float = 1):
    if lower_limit >= upper_limit:
        raise ValueError('lower_limit must be less than upper_limit')
    targets = []
    for i in range(using_data.shape[0]):
        for j in range(using_data.shape[1]):
            if i < j:
                if lower_limit < using_data[i][j] < upper_limit:
                    for z in range(len(targets)):
                        if i in targets[z] or j in targets[z]:
                            targets[z].add(i)
                            targets[z].add(j)
                            break
                    else:
                        targets.append({i, j})
    return targets


def merge_event(using_data, lower_limit: float = 0.8, upper_limit: float = 1):
    if lower_limit >= upper_limit:
        raise ValueError('lower_limit must be less than upper_limit')
    targets = []
    for i in range(using_data.shape[0]):
        for j in range(using_data.shape[1]):
            if i < j:
                if lower_limit < using_data[i][j] < upper_limit:
                    for z in range(len(targets)):
                        if i in targets[z] or j in targets[z]:
                            targets[z].add(i)
                            targets[z].add(j)
                            break
                    else:
                        targets.append({i, j})
    return targets


if __name__ == '__main__':
    u_data = pd.read_csv('./data/embedding_data.csv').values.tolist()
    esm_entity = get_entity_similarity_matrix(u_data)
    esm_event = get_event_similarity_matrix(u_data)
