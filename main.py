import file_manager
import id3
import config

dataset = file_manager.read_dataset()

attributes = dataset.keys().drop(config.file_label)
tree = id3.id3(dataset, attributes)
id3.print_tree(tree)
print(tree)