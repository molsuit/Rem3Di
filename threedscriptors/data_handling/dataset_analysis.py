import numpy as np
from threedscriptors.data_handling.dataset import BaseDataset
import matplotlib.pyplot as plt
import os
import torch 
from itertools import groupby, combinations, chain 
from threedscriptors.data_handling.data_utils import rmsd
from ase.visualize.plot import plot_atoms
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature, get_invariant_indices
from threedscriptors.data_handling.data_utils import get_atom_species_in_smiles, get_all_atom_counts
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator



class DatasetPostLoadAnalysis():


    def __init__(self, dataset : BaseDataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def calculate_atomic_descriptor_norms(atomic_descriptors, padding_masks):
        norms = np.linalg.norm(atomic_descriptors, axis = (0,1), where = ~padding_masks)
        return norms
    
    def plot_molecule_size_distribution(self):
        counts = get_all_atom_counts(self.dataset.molecules, heavy_atoms_only= self.dataset.dataset_config.only_heavy_atoms)

        fig = plt.figure()
        plt.hist(np.array(counts))
        plt.xlabel("Molecule Size")
        plt.ylabel("Frequency")
        fig.savefig(f"{self.output_dir}/histogram_molecule_size.png")
        plt.close(fig)
              
    def get_dataset_size(self):

        size = self.dataset.embeddings.element_size() * self.dataset.embeddings.nelement()

        return size


    def get_invariants_std(self):

        irreps = get_mace_calculator_irrep_signature(self.dataset.dataset_config.embedding_model_config.mace_calc)
        invariant_indices, _ = get_invariant_indices(irreps)


        invariants = self.dataset.embeddings[:,:,invariant_indices]

        std = torch.std(invariants, dim = (0,1))

        plt.figure()
        plt.hist(std)
        plt.savefig(f"{self.output_dir}/invariants_std.png")

        bar_std = plt.figure(figsize=(12, 4), dpi=100)
        plt.bar(np.arange(len(std)),std, width=1,align="edge")
        plt.xlim([0, len(std)])
        plt.xlabel("MACE feature dimension")
        plt.ylabel("Std of each invariant MACE feature dimension")
        bar_std.savefig(f"{self.output_dir}/embeddings_std.png")


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


    def get_atom_species(self):
        smiles_iterator = ListSmilesIterator(self.dataset.smiles_list)
        return get_atom_species_in_smiles(smiles_iterator)

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

 
        assert N_samples == self.dataset.regression_targets.shape[0]
        assert N_samples == self.dataset.regression_masks.shape[0]

        if self.dataset.atomic_positions is not None:
            assert N_samples == self.dataset.atomic_positions.shape[0]
        
        assert N_samples == len(self.dataset.molecules)


        assert (self.dataset.embeddings[self.dataset.padding_mask.unsqueeze(-1).expand_as(self.dataset.embeddings)] == 0).all()

        
    def check_conformer_distance(self):

        if self.dataset.dataset_config.N_conformers == 1:
            return
        
        mol_id_chunks = [list(g) for _, g in groupby(range(len(self.dataset.mol_ids)), key = lambda i : self.dataset.mol_ids[i])]

        global_rmsds = []

        for mol_indices in mol_id_chunks:
            mols =[ self.dataset.molecules[i] for i in mol_indices]
            positions = [m.get_positions() for m in mols]
            

            rmsds = []

            for pos_conf_A, pos_conf_B in combinations(positions,2):
                rmsds.append(rmsd(pos_conf_A, pos_conf_B))

            global_rmsds.append(rmsds)

        

        flattend_rmsds = list(chain.from_iterable(global_rmsds))
        rmsd_fig = plt.figure()

        plt.hist(flattend_rmsds)
        rmsd_fig.savefig(f"{self.output_dir}/rmsd_distribution.png")



    def run(self):

        self.plot_regression_target_distribution()
        counts = self.count_samples_per_task()
        mean, stds = self.mean_and_std()
        size = self.get_dataset_size()
        atom_species = self.get_atom_species()
        self.plot_molecule_size_distribution()
        #self.plot_relaxed_atoms()


        self.check_conformer_distance()
        self.get_invariants_std()

        self.check_dataset_integrity()