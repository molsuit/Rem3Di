import polaris as po 


competition = po.load_competition("asap-discovery/antiviral-admet-2025")

competition.cache()

train, test = competition.get_train_test_split()
print(train.target_cols)

