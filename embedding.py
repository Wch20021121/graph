from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm


# 计算文本嵌入
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


def embeddings_data(org_data, model, tokenizer):
    for i in tqdm(range(len(org_data)), desc='embedding'):
        org_data[i].append(get_embeddings(str(org_data[i][0]), model, tokenizer))
        org_data[i].append(get_embeddings(str(org_data[i][1]), model, tokenizer))
        org_data[i].append(get_embeddings(str(org_data[i][2]), model, tokenizer))
        org_data[i].append(get_embeddings(str(org_data[i][5]), model, tokenizer))
    tar_data = pd.DataFrame(org_data, columns=["text", "eventA", "eventB", "relationship", "years", 'entity',
                                               "text_embedding", "eventA_embedding", "eventB_embedding",
                                               "entity_embedding"])
    tar_data.to_csv('./data/res_data.csv', index=False)

    return tar_data.values.tolist()


if __name__ == '__main__':
    bert_path = r'D:\Model\transformers\text-classier\bert_model\bert-base-chinese'
    # 加载BERT模型和分词器
    tr = BertTokenizer.from_pretrained(bert_path)
    mol = BertModel.from_pretrained(bert_path)
    data = pd.read_csv('./data/original_data.csv').values.tolist()
    # # 获取实体和事件(原事件，事件A和事件B)的文本嵌入
    # for i in tqdm(range(len(data)), desc='embedding'):
    #     data[i].append(get_embeddings(str(data[i][0]), mol, tr))
    #     data[i].append(get_embeddings(str(data[i][1]), mol, tr))
    #     data[i].append(get_embeddings(str(data[i][2]), mol, tr))
    #     data[i].append(get_embeddings(str(data[i][5]), mol, tr))
    # # 保存原数据和相似度数据
    # data = pd.DataFrame(data, columns=["text", "eventA", "eventB", "relationship", "years", 'entity',
    #                                    "text_embedding",
    #                                    "eventA_embedding", "eventB_embedding", "entity_embedding"])
    # data.to_csv('./data/embedding_data.csv', index=False)
    x = embeddings_data(data, mol, tr)
