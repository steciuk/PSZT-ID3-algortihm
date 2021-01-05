import config
import math


def __get_dataset_entropy(dataset):
    entropy = 0
    frequencies = dataset[config.file_label].value_counts(normalize=True)
    for frequency in frequencies:
        entropy += -frequency * math.log(frequency)

    return entropy


def __get_split_entropy(dataset, attribute):
    entropy = 0
    attribute_values = dataset[attribute].unique()
    for value in attribute_values:
        subset = dataset.loc[dataset[attribute] == value]
        entropy += len(subset) / len(dataset) * __get_dataset_entropy(subset)

    return entropy


def __get_inf_gain(dataset, attribute):
    return __get_dataset_entropy(dataset) - __get_split_entropy(dataset, attribute)


def __get_attribute_with_biggest_inf_gain(dataset, attributes):
    max_gain = -math.inf
    dataset_entropy = __get_dataset_entropy(dataset)
    split_attribute = attributes[0]

    for attribute in attributes:
        gain = dataset_entropy - __get_split_entropy(dataset, attribute)
        if gain > max_gain:
            max_gain = gain
            split_attribute = attribute

    return split_attribute


def id3(dataset, attributes):
    classes = dataset[config.file_label]
    node = {"count": len(dataset)}

    # Theres only one class left
    if len(classes.unique()) == 1:
        node["label"] = classes.unique()[0]
        return node

    # There are no more attributes to compute
    if not len(attributes):
        node["label"] = classes.mode()
        return node

    splitting_attribute = __get_attribute_with_biggest_inf_gain(dataset, attributes)

    remaining_attributes = attributes.drop(splitting_attribute)
    node["attribute"] = splitting_attribute
    node["children"] = {}

    for value in dataset[splitting_attribute].unique():
        subset = dataset.loc[dataset[splitting_attribute] == value]
        node["children"][value] = id3(subset, remaining_attributes)

    return node


def print_tree(node):
    def __print_append_tabs(element, num_of_tabs, nl):
        output = ""
        if nl:
            output += "\n"
            for i in range(num_of_tabs):
                output += "\t"

        print(output + element, end="")

    def __print_tree(nodes, depth):
        d = depth

        if "label" in nodes:
            __print_append_tabs(nodes["label"], d, False)

        if "attribute" in nodes:
            __print_append_tabs(nodes["attribute"], d, False)

        __print_append_tabs(" (" + str(nodes["count"]) + ")", d, False)

        if "children" in nodes:
            d += 1
            for ch in nodes["children"]:
                __print_append_tabs(ch + ": ", d, True)
                __print_tree(nodes["children"][ch], d)

    __print_tree(node, 0)
    print()


def predict_class(tree, instance):
    def get_most_frequent_child(children):
        max_count = -math.inf
        most_frequent = None
        for child in children.values():
            if child["count"] > max_count:
                max_count = child["count"]
                most_frequent = child

        return most_frequent

    if "label" in tree:
        return tree["label"]
    elif "attribute" in tree:
        attribute = tree["attribute"]
        instance_value = instance[attribute]
        if instance_value in tree["children"]:
            subtree = tree["children"][instance_value]
            return predict_class(subtree, instance)
        else:
            subtree = get_most_frequent_child(tree["children"])
            return predict_class(subtree, instance)
    else:
        raise Exception("invalid tree!")