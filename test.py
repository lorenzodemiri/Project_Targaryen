import vaex
import pandas as pd
import numpy as np
import gc
from my_func import Transformer



def fill_missing(df, filler):
    df[get_columns()] = df[get_columns()].fillna(filler)
    return df

def get_columns():
    string = 'feature_{}'
    res = []
    for i in range(0, 130):
        res.append(string.format(i))
    return res


df_train = vaex.from_csv("./Datasets/train.csv",
                         convert=True, chunk_size=3_000_000)
val_max = df_train.max(get_columns())
val_min = df_train.min(get_columns())
val_range = val_max - val_min
del val_max
gc.collect()
features = [f'feature_{i}' for i in range(130)]
filler = pd.Series(val_min - 0.01 * val_range, index=features)
train = fill_missing(df_train.to_pandas_df(), filler)
train['action'] = 0
train.loc[train['resp'] > 0.0, 'action'] = 1
train_v = vaex.from_pandas(train)
del train
gc.collect()
s = Transformer(train_v=train_v)
s.get_relevat_feature(100)
