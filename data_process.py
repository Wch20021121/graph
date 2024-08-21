import pandas as pd


def process_data(data: pd.DataFrame):
    # 把所有NaN替换为""
    data = data.fillna("")
    data = data.values.tolist()
    return data
