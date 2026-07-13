from remedi.data_handling.pipelines import reload_dataset_pipeline


def test_log_scaling_data():
    directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_admet_train"
    reload_dataset_pipeline(directory=directory).build()
