from data_process import process_data
from entity import entity_recognition
import pandas as pd
from transformers import BertTokenizer, BertModel
from embedding import embeddings_data
from similarity_calculations import get_entity_similarity_matrix, get_event_similarity_matrix
from create_graph import get_events, merge_node, merge_event
import networkx as nx
from config import data_path, bert_path, color_dict, color_keys, data_ratio


def main():
    # 加载BERT模型和分词器
    tr = BertTokenizer.from_pretrained(bert_path)
    mol = BertModel.from_pretrained(bert_path)
    # 原始数据分布格式如下
    # 事件, 事件A, 事件B, 关系, 年份
    # 事件是一个句子，事件A和事件B是两个句子，关系是事件A和事件B的关系，年份是事件发生的年份

    # 提取实体
    data = process_data(data=pd.read_csv(data_path))
    data = entity_recognition(data)
    print('实体提取完成')

    # 获取实体和事件(原事件，事件A和事件B)的文本嵌入
    data = embeddings_data(data, mol, tr)

    # 相似度计算
    esm_entity = get_entity_similarity_matrix(data)
    esm_event = get_event_similarity_matrix(data)

    # 读取事件列表
    events_list = get_events(data)

    # 根据相似度比例合并事件
    for ratio in data_ratio:
        node_index = len(events_list) * 2 + 2
        event_similarity_dict = merge_event(esm_event, ratio, 1)
        G = nx.DiGraph()

        for i in range(len(events_list)):
            G.add_node(i * 2, name=events_list[i][0], time=events_list[i][3], type='event')
            G.add_node(i * 2 + 1, name=events_list[i][1], time=events_list[i][3], type='event')
            if events_list[i][2] in color_keys:
                G.add_edge(i * 2, i * 2 + 1, relationship=events_list[i][2], color=color_dict[events_list[i][2]])

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


if __name__ == '__main__':
    main()
