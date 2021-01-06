import math
from collections import defaultdict
import numpy as np
import pandas as pd
import config
import id3

K_COLUMNS = ["data_size", "k", "TP", "TN", "FP", "FN", "accuracy", "global_seed", "num_of_reruns"]
COLUMNS = ["train_size", "test_size", "TP", "TN", "FP", "FN", "accuracy", "global_seed", "num_of_reruns"]


def validation_of_full_set_multirun_for_different_dataset_size(dataset, starting_set_part, min_set_part):
    full_results = pd.DataFrame(columns=COLUMNS)
    dataset_size = len(dataset)
    part = starting_set_part
    while part >= min_set_part:
        print("part: " + str(part))
        results = pd.DataFrame(columns=["TP", "TN", "FP", "FN"])
        training_size = round(part * dataset_size)
        for i in range(config.num_of_reruns):
            print("rerun: " + str(i))
            training = dataset.sample(frac=part, random_state=config.rng_seed + i)
            attributes = training.keys().drop(config.file_label)
            tree = id3.id3(training, attributes)
            results = results.append(__test(tree, dataset))

        dataframe = __build_final_dataframe_for_full_validation(results, training_size, dataset_size)
        full_results = full_results.append(dataframe, ignore_index=True)
        part /= 2

    return full_results


def k_validation_multirun_for_different_dataset_size(dataset, k, starting_set_part, min_set_part):
    full_results = pd.DataFrame(columns=K_COLUMNS)
    part = starting_set_part
    while part > min_set_part:
        print("part: " + str(part))
        dt = dataset.sample(frac=part, random_state=config.rng_seed)
        data = __k_multirun(dt, k)
        full_results = full_results.append(data, ignore_index=True)
        part /= 2

    return full_results


def k_validation_multirun_for_different_k(dataset, k_min, k_max):
    full_results = pd.DataFrame(columns=K_COLUMNS)
    for k in range(k_min, k_max + 1):
        print("k: " + str(k))
        data = __k_multirun(dataset, k)
        full_results = full_results.append(data, ignore_index=True)

    return full_results


def __build_final_dataframe_for_full_validation(results, training_size, test_size):
    results = results.mean()
    accuracy = (results["TP"] + results["TN"]) / test_size
    return pd.DataFrame(data=[
        [training_size, test_size, results["TP"], results["TN"], results["FP"], results["FN"], accuracy,
         config.rng_seed,
         config.num_of_reruns]], columns=COLUMNS)


def __k_multirun(dataset, k):
    dataset_size = len(dataset)
    results = pd.DataFrame(columns=["TP", "TN", "FP", "FN"])
    for i in range(config.num_of_reruns):
        print("rerun: " + str(i))
        dt = dataset.sample(frac=1, random_state=config.rng_seed + i)
        results = results.append(__k_fold(dt, k), ignore_index=True)

    return __build_final_dataframe_for_k_validation(results, k, dataset_size)


def __build_final_dataframe_for_k_validation(results, k, dataset_size):
    results = results.mean()
    accuracy = (results["TP"] + results["TN"]) / dataset_size
    return pd.DataFrame(data=[
        [dataset_size, k, results["TP"], results["TN"], results["FP"], results["FN"], accuracy, config.rng_seed,
         config.num_of_reruns]], columns=K_COLUMNS)


def __k_fold(dataset, k):
    split_dataset = np.array_split(dataset, k)
    results = pd.DataFrame(columns=["TP", "TN", "FP", "FN"])
    for i in range(k):
        train_set = split_dataset.copy()
        test_set = split_dataset[i]
        del train_set[i]
        train = pd.concat(train_set, sort=False)

        attributes = train.keys().drop(config.file_label)
        tree = id3.id3(train, attributes)
        tmp_results = __test(tree, test_set)
        results = results.append(tmp_results, ignore_index=True)

    results = results.sum()
    return results


def __test(tree, test_set):
    results = defaultdict(int)
    results["TP"] = 0
    results["TN"] = 0
    results["FP"] = 0
    results["FN"] = 0
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
