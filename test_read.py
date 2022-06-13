import numpy as np
from collections import Counter
import pickle

if __name__ == "__main__":

    with open('./dataset/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)

    all_data = res['data']
    all_label = res['label']
    print(Counter(all_label))
    data1 = all_data[0]
    print(data1)