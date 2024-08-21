# 数据地址
data_path = './data/original_data.csv'
# 原始数据分布格式如下
# 事件, 事件A, 事件B, 关系, 年份
# 事件是一个句子，事件A和事件B是两个句子，关系是事件A和事件B的关系，年份是事件发生的年份

# BERT模型地址-默认使用bert-base-chinese
bert_path = r'D:\Model\transformers\text-classier\bert_model\bert-base-chinese'

# LTP模型地址
ltp_model_path = r"D:\Model\LTP_model"

# 颜色dict,根据你的关系和颜色喜好对应
color_dict = {'顺承': "red", '上下位': "blue", '条件': "yellow", '内容关系': "black", '并发': "green",
              '因果': "purple"}

color_keys = ['顺承', '上下位', '条件', '内容关系', '并发', '因果']

# 数据比例列表结果数据集,用于事件合并的相似度比例
data_ratio = [0.943]

# neo4j数据库地址,用户名和密码
neo4j_username = "neo4j"
neo4j_password = "Wch20021121"
neo4j_url = "bolt://localhost:7687"


# 故事生成模块数据地址, 先运行mian生成该数据
story_data_path = './data/res_data.csv'

# LLM模型API_KEY，类型
agicto_api_key = 'sk-ZUzP9hLRDl96oE8broZErsFd4Yw8morsP9Xp7EjACFeJmDt3'
tongyi_api_key = 'sk-87a8010d70854b1589b157d6823a62d3'
agicto_api_base = "https://api.agicto.cn/v1"
llm_model = 'gpt-4'
