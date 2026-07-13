from remedi.data_handling.dataset_concatenation import DatasetConcatenation
from remedi.data_handling.dataset_io import (
    load_data_from_disk,
    store_data_to_disk,
)


def test_dataset_concatenation():
    adme_fang_dataset = load_data_from_disk(
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/adme_fang",
        load_molecules=True,
    )
    admet_antiviral = load_data_from_disk(
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/antiviral_admet",
        load_molecules=True,
    )

    cmrt = load_data_from_disk(
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt",
        load_molecules=True,
    )

    concatenation = DatasetConcatenation(
        datasets=[adme_fang_dataset, admet_antiviral, cmrt]
    )

    new_dataset = concatenation.concatenate()

    assert len(
        set([dicts.values() for dicts in concatenation.collected_relabel_dicts])
    ) == len(
        [dicts.values() for dicts in concatenation.collected_relabel_dicts]
    )  # Checks if the relabling of the mol ids are unique

    store_data_to_disk(
        new_dataset,
        "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/multitask_dataset",
    )


# test_dataset_concatenation()
