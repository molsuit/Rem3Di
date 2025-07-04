import numpy as np
from threedscriptors.data_handling.dataset import BaseDataset
import matplotlib.pyplot as plt
import os

from ase.visualize.plot import plot_atoms


class DatasetPostLoadAnalysis():


    def __init__(self, dataset : BaseDataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def calculate_atomic_descriptor_norms(atomic_descriptors, padding_masks):
        norms = np.linalg.norm(atomic_descriptors, axis = (0,1), where = ~padding_masks)
        return norms

    def count_samples_per_task(self):
        num_samples = self.dataset.regression_masks.sum(0).tolist()
        samples_per_task = dict(zip(self.dataset.dataset_config.get_task_names(),num_samples))
        return samples_per_task

    def mean_and_std(self):
        task_names = self.dataset.dataset_config.get_task_names()
        
        rt = self.dataset.regression_targets.cpu().numpy()
        rm = self.dataset.regression_masks.bool().cpu().numpy()

        means = np.mean(rt, axis = 0, where = rm)
        stds = np.std(rt, axis = 0, where = rm)

        return dict(zip(task_names, means)), dict(zip(task_names, stds))
    

    def plot_relaxed_atoms(self):
        
        N_horizontal = 3
        N_vertical = (len(self.dataset.molecules) // 3 )+1

        fig, axarr = plt.subplots(N_vertical, N_horizontal)
        
        fig.set_figheight(4*N_vertical)
        fig.set_figwidth(4*N_horizontal)

        for i, mol in enumerate(self.dataset.molecules):
            plot_atoms(mol, axarr[i // 3 , i % 3])

        fig.savefig(f"{self.output_dir}/relaxed_atoms.png")



    def plot_regression_target_distribution(self):
        
        for idx, task in enumerate(self.dataset.dataset_config.tasks):
            regression_targets = self.dataset.regression_targets[:,idx]
            regression_masks = self.dataset.regression_masks[:,idx]
            
            y = regression_targets[regression_masks.bool()]

            fig = plt.figure()
            plt.hist(y)
            fig.savefig(f"{self.output_dir}/distribution_{task.task_name}_labels.png")
            plt.close(fig)

            log_scaled_y = np.log(y[y>0])
            fig = plt.figure()
            plt.hist(log_scaled_y)
            plt.savefig(f"{self.output_dir}/distribution_{task.task_name}_log_scaled_labels.png")


    def check_dataset_integrity(self):

        N_samples = self.dataset.dataset_config.N_molecules

        assert self.dataset.embeddings.shape[0] == N_samples
        assert N_samples == self.dataset.padding_mask.shape[0]

        print(N_samples) 
        print(self.dataset.regression_targets.shape)
        assert N_samples == self.dataset.regression_targets.shape[0]
        assert N_samples == self.dataset.regression_masks.shape[0]

        if self.dataset.atomic_positions is not None:
            assert N_samples == self.dataset.atomic_positions.shape[0]
        
        assert N_samples == len(self.dataset.molecules)


        assert (self.dataset.embeddings[self.dataset.padding_mask.unsqueeze(-1).expand_as(self.dataset.embeddings)] == 0).all()
        
        



    def run(self):

        self.plot_regression_target_distribution()
        counts = self.count_samples_per_task()
        mean, stds = self.mean_and_std()
        #self.plot_relaxed_atoms()
        print(counts)
        print(mean)
        print(stds)

        self.check_dataset_integrity()