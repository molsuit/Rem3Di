
    #shared_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
#
#
#
    #rows, cols = torch.tril_indices(row=len(task_names), col=len(task_names), offset=-1)
#
#
    #task_pair_indices = torch.nonzero(torch.tril(torch.ones((len(task_names),len( task_names))),-1))
#
    #map_task_to_sim_dict = {idx : (task_names[p[0]],task_names[p[1]]) for idx, p in enumerate(task_pair_indices)}
#
    #full_training_grad_alignment = []
#
#
#
#
            



    #
    #
    #print(map_task_to_sim_dict)
    #
    #
    #cosine_sim_dict = {v : [] for v in map_task_to_sim_dict.values()}
    #print(cosine_sim_dict)
    #
    #
    #full_training_grad_alignment = torch.stack(full_training_grad_alignment)
    #print(full_training_grad_alignment.shape)
    #
    #
    #import matplotlib.pyplot as plt
    #
    #
    #
    #times = list(range(training_config.epochs))
    #
    #for i, task_pair in map_task_to_sim_dict.items():
    #
    #
    #    task_pair_data= full_training_grad_alignment[:,i,:].squeeze()
    #    
    #
    #    print(task_pair_data.shape)
    #    print(times)
    #
    #    fig, ax = plt.subplots()
    #
    #    fig.set_figwidth(20.)

    #    ax.violinplot(task_pair_data.T, positions=times, widths=0.8, #showmeans=False, showextrema=True, showmedians=True)
    #    ax.set_xlabel("Time")
    #    ax.set_ylabel("Cosine Similarity of gradients")
    #    ax.set_title(f"Distribution grad cosine sim over training time {task_pair}")

    #    fig.savefig(f"{training_config.training_data_dir}/grad_tasks_cosine_sim_{str#(task_pair)}.png")
    #
    #
    #    print(task_pair)
    #    ratio_0 = (task_pair_data < 0.0).float().mean()
    #    ratio_0_1 = (task_pair_data < -0.1).float().mean() 
    #    print(f"Smaller 0.0  {ratio_0}")
    #    print(f"Smaller -0.1  {ratio_0_1}")
