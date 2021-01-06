import file_manager
from validation import k_validation_multirun_for_different_dataset_size

dataset = file_manager.read_dataset()
results = k_validation_multirun_for_different_dataset_size(dataset, 3, 1.0, 0.001)
print(results)
file_manager.write_to_csv(results, "ex2.csv")