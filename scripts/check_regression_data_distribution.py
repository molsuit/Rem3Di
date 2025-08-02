import polaris as po

# Load the dataset from the Hub
dataset = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")

# Get information on the dataset size
dataset.size()

# Or, similarly:
dataset.load_to_memory()

print(dataset._zarr_data.keys())

tasks= ['HLM', 'KSOL', 'LogD', 'MDR1-MDCKII', 'MLM']

import matplotlib.pyplot as plt
import numpy as np

for task in tasks:
    a = dataset._zarr_data[task]

    m = np.nanmean(a)
    s = np.nanstd(a)
    print(f"T {task} : mean {m}, std :{s}")

    a = a[~np.isnan(a)]
    print(a[a<=0])
    a = a[a>0]
    la = np.log(a)
    fig = plt.figure()
    plt.hist(a)
    plt.savefig(f"{task}_dist.png")

    fig = plt.figure()
    plt.hist(la)
    plt.savefig(f"{task}_log_dist.png")
