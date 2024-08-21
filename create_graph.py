import numpy as np
from tqdm import tqdm
import pandas as pd
from similarity_calculations import merge_event
import networkx as nx


def get_events(original_events):
    targets = []
    for event in original_events:
        targets.append([event[1], event[2], event[3], event[4]])
    return targets


# node3是新的节点
def merge_node(graph, node1, node2, node3):
    # 添加一个新的节点
    node_id, node_data = node3[0], node3[1]
    graph.add_node(node_id, **node_data)

    # 判断如果存在关系则删除，否则就跳过,删除node1，node2的关系
    if graph.has_edge(node1, node2):
        graph.remove_edge(node1, node2)

    if find_node_by_id(graph, node1):
        for predecessor in graph.predecessors(node1):
            graph.add_edge(predecessor, node_id, **graph.get_edge_data(predecessor, node1))
        for successor in graph.successors(node1):
            graph.add_edge(node_id, successor, **graph.get_edge_data(node1, successor))
        graph.remove_node(node1)

    if find_node_by_id(graph, node2):
        for predecessor in graph.predecessors(node2):
            graph.add_edge(predecessor, node_id, **graph.get_edge_data(predecessor, node2))
        for successor in graph.successors(node2):
            graph.add_edge(node_id, successor, **graph.get_edge_data(node2, successor))
        graph.remove_node(node2)

    # 返回新的图
    return graph


# 通过id找到节点
def find_node_by_id(graph, node_id):
    if node_id in graph.nodes:
        return True
    else:
        return False


if __name__ == '__main__':
    # 读取事件相似度矩阵
    esm_event = np.load('data/matrix/event_similarity_matrix.npy')
    # 读取事件列表
    emb_data = pd.read_csv('./data/res_data.csv').values.tolist()
    events_list = get_events(emb_data)
    # 颜色dict
    color_dict = {'顺承': "red", '上下位': "blue", '条件': "yellow", '内容关系': "black", '并发': "green",
                  '因果': "purple"}
    color_keys = ['顺承', '上下位', '条件', '内容关系', '并发', '因果']
    # 数据比例列表结果数据集
    data_ratio = [0.931, 0.932, 0.933, 0.934, 0.935, 0.936, 0.937, 0.938, 0.939, 0.94, 0.941, 0.942, 0.943, 0.944,
                  0.945, 0.946, 0.947, 0.948, 0.949]
    for ratio in data_ratio:
        node_index = len(events_list) * 2 + 2
        event_similarity_dict = merge_event(esm_event, ratio, 1)
        G = nx.DiGraph()
        for i in range(len(events_list)):
            G.add_node(i * 2, name=events_list[i][0], time=events_list[i][3], type='event')
            G.add_node(i * 2 + 1, name=events_list[i][1], time=events_list[i][3], type='event')
            if events_list[i][2] in color_keys:
                G.add_edge(i * 2, i * 2 + 1, relationship=events_list[i][2], color=color_dict[events_list[i][2]])
        # nx.write_gml(G, './result/graph_0.gml'.format(ratio))
        for i in range(len(event_similarity_dict)):
            event_similarity_dict[i] = list(event_similarity_dict[i])
            if event_similarity_dict[i][0] % 2 == 0:
                new_index = 0
            else:
                new_index = 1
            # 构建新的节点
            new_node = [node_index, {'name': events_list[event_similarity_dict[i][0] // 2][new_index],
                                     'time': events_list[event_similarity_dict[i][0] // 2][3],
                                     'type': 'event'}]
            # 第一个事件节点肯定存在
            node_one = event_similarity_dict[i][0]
            node_index += 1

            # 合并0和剩下所有的节点
            for j in range(1, len(event_similarity_dict[i])):
                node_two = event_similarity_dict[i][j]
                # 合并节点
                G = merge_node(G, node_one, node_two, new_node)
                node_one = new_node[0]
                node_index += 1
                new_node[0] = node_index
            else:
                pass
        # 保存图
        nx.write_gml(G, './result/graph_{}.gml'.format(ratio))
