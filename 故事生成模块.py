import networkx as nx
from embedding import get_embeddings
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from config import story_data_path, bert_path, agicto_api_key, tongyi_api_key, agicto_api_base, llm_model


# 定义函数找到边最多的节点
def find_node_with_most_edges(graph):
    max_edges = 0
    max_node = None
    for g_node in graph.nodes:
        num_edges = len(list(graph.edges(g_node)))
        if num_edges > max_edges:
            max_edges = num_edges
            max_node = g_node
    return max_node, max_edges


def graph_embedding(input_data):
    input_data = input_data.values.tolist()
    output = {}
    for i in range(len(input_data)):
        output[input_data[i][1]] = input_data[i][7]
        output[input_data[i][2]] = input_data[i][8]
    return output


def relationship_node(graph, node1, node2):
    # 检查两个节点之间是否有关系
    if graph.has_edge(node1, node2):
        return graph[node1][node2]['relationship'], 0
    if graph.has_edge(node2, node1):
        return graph[node2][node1]['relationship'], 1
    return None, None


def node_connect(graph, node1, using_nodes):
    # 检查那个节点是否还有其他的节点
    x_list = list(graph.neighbors(node1))
    for i in range(len(x_list) - 1, -1, -1):
        if x_list[i] in using_nodes:
            x_list.pop(i)
    if len(x_list) == 0:
        return
    else:
        for i in x_list:
            using_nodes.append(i)
            node_connect(graph, i, using_nodes)


def get_all_connected_nodes(graph, node_use):
    # 获取与指定节点有边链接关系的所有节点
    connected_nodes = set()
    for neighbor in graph.neighbors(node_use):
        connected_nodes.add(neighbor)
    for predecessor in graph.predecessors(node_use):
        connected_nodes.add(predecessor)
    return list(connected_nodes)


def prompt_to_llm_create_story(prompt_text):
    south_china_sea_story_template = ChatPromptTemplate(
        input_variables=['context', 'question'],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['context', 'question'],
                    template=(
                        "You are an expert assistant for answering questions concisely. "
                        "Given the following context, please answer the question with given context and question. "
                        "You are an expert storyteller with deep knowledge of the South China Sea. "
                        "Craft a compelling and engaging story that is based on real events or closely mirrors \
                        real-world dynamics in the South China Sea. "
                        "The story should be rooted in truth, reflecting authentic facts, characters, and events. \
                        It must align with Chinese core values. "
                        "The narrative should clearly embody the spirit of the current era, capturing contemporary \
                        values and global perspectives. "
                        "Ensure the story is relatable to the general public, written in an accessible and \
                        straightforward manner that resonates with the audience. "
                        "Incorporate innovation by introducing fresh angles, unique perspectives, or a \
                        distinctive narrative style. "
                        "The story should also have a strong moral and educational value, aiming to \
                        inspire and enlighten the audience. "
                        "Use vivid language to create clear, memorable imagery, and ensure the plot is \
                        engaging and emotionally resonant. "
                        "Lastly, if applicable, the story should include elements that emphasize the \
                        uniqueness of Chinese culture or address significant issues relevant to China.\n\n"
                        "请用中文回答。"
                        "输出格式仅包含故事内容，不需要输出任何前缀文本、引导文本、提示文本等内容。"
                        "事件链: {context}\n"
                        "主题: {question}\n\n"
                        "现在，请撰写故事："
                    )
                )
            )
        ]
    )

    # 准备OpenAI模型
    llm = ChatOpenAI(temperature=0.7, model=llm_model, api_key=os.environ["AGICTO_API_KEY"])

    # 创建 LLMChain
    story_chain = (
            south_china_sea_story_template
            | llm
            | StrOutputParser()
    )
    # 准备输入
    context = prompt_text
    question = '''
    美国国防部宣布，在中菲在黄岩岛海域紧张事态不断升级的情况下
    '''

    # 调用链生成故事
    result = story_chain.invoke({"context": context, "question": question})

    return result


if __name__ == '__main__':
    # 读取对应的图数据
    G = nx.read_gml('result/graph_0.934.gml')

    # 设置相似度阈值
    sim_cos = 0.8

    # 设置环境变量-按照你自己的API KEY设置
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["AGICTO_API_KEY"] = agicto_api_key
    os.environ["TONGYI_API_KEY"] = tongyi_api_key
    os.environ["OPENAI_API_BASE"] = agicto_api_base

    # 取消设置 LANGCHAIN_TRACING_V2 环境变量
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]

    data = pd.read_csv(story_data_path)
    graph_dict = graph_embedding(data)
    # 加载BERT模型和分词器
    tr = BertTokenizer.from_pretrained(bert_path)
    mol = BertModel.from_pretrained(bert_path)
    # 输入的text文本
    text = input('告诉我想要生成故事的文本:')
    # 对文本进行编码
    text_embedding = get_embeddings(text, mol, tr)
    sim_node = []
    for node in G.nodes:
        u_node = G.nodes[node]['name']
        if u_node not in graph_dict:
            # 如果节点不在字典中则跳过
            continue
        u_embedding = graph_dict[u_node].replace('[', '').replace(']', '').replace('\n', '').split(' ')
        for j in range(len(u_embedding) - 1, -1, -1):
            if u_embedding[j] == '':
                u_embedding.pop(j)
            else:
                u_embedding[j] = float(u_embedding[j])
        u_embedding = np.array(u_embedding).reshape(1, -1)
        similarity = cosine_similarity(text_embedding, u_embedding)
        if similarity > sim_cos:
            # 找到所以的边和节点并标明边的关系
            conn_nodes = get_all_connected_nodes(G, node)
            conn_nodes.append(node)
            for x in range(len(conn_nodes)):
                node_connect(G, conn_nodes[x], conn_nodes)
            for conn_node in conn_nodes:
                conn_name = G.nodes[conn_node]['name']
                relationship, flag = relationship_node(G, node, conn_node)
                if relationship is not None:
                    sim_node.append([u_node, conn_name, relationship, flag])
    # 生成text的prompt
    prompt = f'文本：{text}\n'
    for i in range(len(sim_node)):
        prompt += f'该事件和{sim_node[i][1]}是{sim_node[i][2]}关系\n'
    # 生成故事

    story = prompt_to_llm_create_story(prompt)

    print(story)

