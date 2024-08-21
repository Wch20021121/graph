from ltp import LTP
import pandas as pd
from data_process import process_data
from config import ltp_model_path


def entity_recognition(or_data):
    ltp = LTP(pretrained_model_name_or_path=ltp_model_path)
    # LTP分词
    for i in range(len(or_data)):
        text = or_data[i][1]
        output = ltp.pipeline(text, tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
        entities = []
        for j in range(len(output['ner'])):
            entity = output['ner'][j][1]
            if entity != 'O' and entity not in entities:
                entities.append(entity)
        or_data[i].append(entities)
    # 保存数据
    tar_data = pd.DataFrame(or_data, columns=["years", "text", "place", "nature", "filed", "entities"])
    # tar_data.to_csv("./data/ner_data.csv", index=False)
    return tar_data.values.tolist()


if __name__ == "__main__":
    # 读取数据
    data_path = "./data/南海事理图谱总数据集_标签版.xlsx"
    data = pd.read_excel(data_path)
    data = process_data(data)
    # ltp = LTP(pretrained_model_name_or_path="./LTP_model")
    # # LTP分词
    # for i in range(len(data)):
    #     text = data[i][1]
    #     output = ltp.pipeline(text, tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
    #     entities = []
    #     for j in range(len(output['ner'])):
    #         entity = output['ner'][j][1]
    #         if entity != 'O' and entity not in entities:
    #             entities.append(entity)
    #     data[i].append(entities)
    # # 保存数据
    # data = pd.DataFrame(data, columns=["years", "text", "place", "nature", "filed", "entities"])
    # data.to_csv("./data/ner_data.csv", index=False)
    result = entity_recognition(data)
