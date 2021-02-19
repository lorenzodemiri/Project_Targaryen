import numpy as np
from six import reraise
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from IPython.display import display
import pandas as pd
import vaex
import gc
import time

def get_columns(string):
    res = []
    for i in range(0, 130):
        res.append(string.format(i))
    return res
def get_fort_text(string, i):
    return string.format(i)

class Transformer:
    def __init__(self, train_v):
        self.train_v = train_v
    
    def feature_transform(self, train_v):
       
        print("Loading First Features\n")
        for fet, sh, log, sqrt in zip(get_columns('feature_{}'), get_columns('shift_feature_{}'), get_columns('log_feature_{}'), get_columns('sqrt_feature_{}')):
            val_min = train_v.min(fet)
            train_v[sh] = train_v[fet] + abs(val_min + 1).astype(np.float16)
            train_v[log] = np.log(train_v[sh])
            train_v[sqrt] = np.sqrt(train_v[sh])
            print(".",end='')
        train_v.drop(get_columns('shift_feature_{}'), inplace=True)
        
        time.sleep(10)
        print("\nLoading Second Features\n")
        # From the Shap Dependence plots, the following features seem to have cubic relationship with target
        cubic = [37, 39, 67, 68, 89, 98, 99, 118, 119, 121, 124, 125, 127]
        for i in cubic:
            threes = np.full(len(train_v), 3)
            string = 'cub_feature_{}'.format(i)
            string_fe = 'feature_{}'.format(i)
            arr = np.power(train_v.to_arrays(
                column_names=string_fe), threes)
            temp_v =  vaex.from_arrays(x = arr)
            train_v[string] = temp_v[:,0]
            print(".", i, end='')
        time.sleep(10)

        print("\nLoading Third Features\n")
        # From the Shap Dependence plots, the following features seem to have quadratic relationship with target
        quad = [6, 37, 39, 40, 53, 60, 61, 62, 63, 64, 67, 68, 89,
                98, 99, 101, 113, 116, 118, 119, 121, 123, 124, 125, 127]
        for i in quad:
            train_v[get_fort_text('quad_feature_{}', i)] = np.square(
                train_v[get_fort_text('feature_{}', i)])
            print(".", i, end='')
        
        return train_v

    def manipulate(self, train_v):

        print("\nAdding Pair Features\n")
        add_pairs = [(3, 6), (15, 26), (19, 26), (30, 37),
                    (34, 33), (35, 39), (94, 65), (101, 4)]
        for i, j in add_pairs:
            train_v["add_{}_{}".format(i,j)] = train_v[get_fort_text(
                'feature_{}', i)] + train_v[get_fort_text('feature_{}', j)]
            train_v["sub_{}_{}".format(i, j)] = train_v[get_fort_text(
                'feature_{}', i)] - train_v[get_fort_text('feature_{}', j)]
            print(".", i, " ", j, end='')

        print("\nAdding Log Pair Features\n")
        add_log_pairs = [(9, 20), (22, 37), (28, 39), (29, 25), (65, 91),
                        (74, 103), (99, 126), (109, 7), (111, 87), (112, 97), (118, 112)]
        for i, j in add_log_pairs:
            train_v["add_{}_log{}".format(i, j)] = train_v[get_fort_text(
                'feature_{}', i)] + train_v[get_fort_text('log_feature_{}', j)]
            train_v["sub_{}_log{}".format(i, j)] = train_v[get_fort_text(
                'feature_{}', i)] - train_v[get_fort_text('log_feature_{}', j)]
            print(".",i," ",j, end='')

        print("\nAdding Mul Features\n")
        mul_pairs = [(5, 42), (12, 66), (37, 45), (39, 95), (122, 35)]
        for i, j in mul_pairs:
            train_v["mul_{}_{}".format(i, j)] = train_v[get_fort_text(
                'feature_{}', i)] * train_v[get_fort_text('feature_{}', j)]
            print(".", i, " ", j, end='')

        print("\nAdding Mul Log Features\n")
        mul_log_pairs = [(5, 42), (6, 42), (11, 99), (21, 42),
                        (81, 66), (98, 20), (122, 35)]
        for i, j in mul_log_pairs:
            train_v["mul_{}_log{}".format(i, j)] = train_v[get_fort_text(
                'feature_{}', i)] *  train_v[get_fort_text('log_feature_{}', j)]
            print(".", i, " ", j, end='')

        return train_v

    def get_relevat_feature(self, i):
        temp = self.train_v[self.train_v.date > 200]
        temp = temp[temp.date < 300]
        temp = self.feature_transform(temp)
        temp = self.manipulate(temp)
        time.sleep(20)
        feature = temp.get_column_names()
        print(feature)
        time.sleep(30)
        for i in ['action', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']:
             feature.remove(i)
        
        selector = SelectKBest(f_classif, k=i)
        time.sleep(10)
        print("selector done")
        X = temp.to_arrays(column_names=feature)
        y = self.train_v.to_arrays(column_names='action')
        print(X8)
        temp_x = selector.fit_transform(X, y)
        time.sleep(20)
        print("Fitted")
        df_new = pd.DataFrame(selector.inverse_transform(temp_x), columns=feature)
        time.sleep(20)
        print("NEW")
        selected_features = df_new.columns[df_new.var() != 0]
        print("OKKKKKK", selected_features)
        return selected_features



