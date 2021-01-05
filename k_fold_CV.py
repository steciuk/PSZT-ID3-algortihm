from collections import defaultdict
import numpy as np
import pandas as pd
import config
import id3

COLUMNS = ['data_size', 'k', "TP", "TN", "FP", "FN", "accuracy", "global_seed", "num_of_reruns"]


def multirun_for_different_dataset_size(dataset, k, min_set_part, max_set_part, step):
    full_results = pd.DataFrame(columns=COLUMNS)
    part = min_set_part
    while part < 1.0 and part <= max_set_part:
        print("part: " + str(part))
        dt = dataset.sample(frac=part, random_state=config.rng_seed)
        data = __multirun(dt, k)
        full_results = full_results.append(data, ignore_index=True)
        part += step

    return full_results

def multirun_for_different_k(dataset, k_min, k_max):
    full_results = pd.DataFrame(columns=COLUMNS)
    for k in range(k_min, k_max + 1):
        print("k: " + str(k))
        data = __multirun(dataset, k)
        full_results = full_results.append(data, ignore_index=True)

    return full_results


def __multirun(dataset, k):
    results = pd.DataFrame(columns=["TP", "TN", "FP", "FN"])
    for i in range(config.num_of_reruns):
        print("rerun: " + str(i))
        dt = dataset.sample(frac=1, random_state=config.rng_seed + i)
        results = results.append(__k_fold(dt, k))

    results = results.mean()
    accuracy = (results["TP"] + results["TN"]) / len(dataset)
    dataframe = pd.DataFrame(data=[
        [len(dataset), k, results["TP"], results["TN"], results["FP"], results["FN"], accuracy, config.rng_seed,
         config.num_of_reruns]], columns=COLUMNS)

    return dataframe


def __k_fold(dataset, k):
    results = defaultdict(int)
    results["TP"] = 0
    results["TN"] = 0
    results["FP"] = 0
    results["FN"] = 0
    split_dataset = np.array_split(dataset, k)
    for i in range(k):
        train_set = split_dataset.copy()
        test_set = split_dataset[i]
        del train_set[i]
        train = pd.concat(train_set, sort=False)

        attributes = train.keys().drop(config.file_label)
        tree = id3.id3(train, attributes)

        for index, row in test_set.iterrows():
            predicted_class = id3.predict_class(tree, row)
            actual_class = row[config.file_label]
            if predicted_class == actual_class:
                if predicted_class == config.positive_class:
                    results["TP"] += 1
                else:
                    results["TN"] += 1
            else:
                if predicted_class == config.positive_class:
                    results["FP"] += 1
                else:
                    results["FN"] += 1

    return pd.DataFrame(data=[[results["TP"], results["TN"], results["FP"], results["FN"]]],
                        columns=["TP", "TN", "FP", "FN"])
