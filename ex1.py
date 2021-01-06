import file_manager
from validation import k_validation_multirun_for_different_k

dataset = file_manager.read_dataset()
results = k_validation_multirun_for_different_k(dataset, 2, 7)
print(results)
file_manager.write_to_csv(results, "ex1.csv")