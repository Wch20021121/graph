from py2neo import Graph, Node, Relationship
import networkx as nx
from 故事生成模块 import find_node_with_most_edges
from config import neo4j_password, neo4j_url, neo4j_username


def create_noe4j_graph(net_graph, noe4j_graph):
    # 创建节点
    node_map = {}
    for node in net_graph.nodes:
        new_node = Node("Entity", name=net_graph.nodes[node]['name'])
        noe4j_graph.create(new_node)
        node_map[net_graph.nodes[node]['name']] = new_node

    # 创建边
    for edge in net_graph.edges:
        start_node = node_map[net_graph.nodes[edge[0]]['name']]
        end_node = node_map[net_graph.nodes[edge[1]]['name']]
        new_edge = Relationship(start_node, net_graph.edges[edge]['relationship'], end_node)
        noe4j_graph.create(new_edge)


if __name__ == '__main__':
    # 读取你想要生成neo4j图的图
    G = nx.read_gml('./result/graph_0.934.gml')
    # 中心节点数据要筛选——找到最具代表性的节点
    G.nodes[find_node_with_most_edges(G)[0]]['name'] = '中国就南海问题发表声明'
    # 连接到 Neo4j 数据库
    graph = Graph(neo4j_url, auth=(neo4j_username, neo4j_password))
    # 清空数据库
    graph.delete_all()
    # 创建图
    create_noe4j_graph(G, graph)
