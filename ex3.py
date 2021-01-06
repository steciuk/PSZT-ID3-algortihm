import file_manager
from validation import validation_of_full_set_multirun_for_different_dataset_size

dataset = file_manager.read_dataset()
results = validation_of_full_set_multirun_for_different_dataset_size(dataset, 1.0, 0.0001)
print(results)
file_manager.write_to_csv(results, "ex3.csv")