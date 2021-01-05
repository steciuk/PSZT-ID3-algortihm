import pandas as pd
import numpy as np
import config
import id3
from k_fold_CV import multirun_for_different_k
from k_fold_CV import multirun_for_different_dataset_size

dataset = pd.read_csv(config.file_name, delimiter=config.file_delimiter)
attributes = dataset.keys().drop(config.file_label)

print(multirun_for_different_dataset_size(dataset, 2, 0.1, 0.6, 0.3))

# for index, row in test_dataset.iterrows():
#     print(id3.predict_class(tree, row))
