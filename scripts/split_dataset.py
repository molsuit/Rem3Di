from threedscriptors.data_handling.dataset_io import store_data_to_disk, load_data_from_disk

dataset_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_test"
dataset = load_data_from_disk(f"{dataset_directory}_full")


splitting_ratio = [0.8,0.2]

split_datasets = dataset.split_dataset(splitting_ratio)

store_data_to_disk(split_datasets[0], f"{dataset_directory}_train")
store_data_to_disk(split_datasets[1], f"{dataset_directory}_valid")
