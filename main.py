import torch 
from model.model import Transformer, TransformerDecoder, TransformerEncoder

import json 

from torch import nn, optim
from model.training import train_loop, validation_loop
from torch.utils.data import random_split, DataLoader

from dataset import AtomEmbeddingDataset

MODEL_DIR = "/home/steffen/projects/mol_descriptors/transformer_model"
DATA_DIR = "/home/steffen/projects/mol_descriptors/data"


architecture_parameter = { 
    "input_dim": 256,
    "num_heads": 8,
    "dim_feedforward": 200,
    "embedding_dim" : 256
}

hyperparameter = {"batch_size":16,"masking_probability": 0.15,"epochs":100,"learning_rate":1e-3}


dataset = AtomEmbeddingDataset.reload_from_disk(f"{DATA_DIR}/descriptor_data.npy",f"{DATA_DIR}/descriptor_mask.npy")

training_data, validation_data, test_data = random_split(dataset,[0.8,0.1,0.1])

training_loader = DataLoader(training_data,batch_size= hyperparameter["batch_size"],shuffle=True,drop_last=True)
validation_loader = DataLoader(training_data,batch_size= hyperparameter["batch_size"],shuffle=True,drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TransformerEncoder(num_layers=2, **architecture_parameter)
decoder = TransformerDecoder(num_layers=2, **architecture_parameter)
model = Transformer(encoder=encoder, decoder=decoder).to(device)


num_opt_steps = hyperparameter["epochs"] * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=hyperparameter["learning_rate"])
# TODO: Warmup / LR Scheduler


for epoch in range(hyperparameter["epochs"]):
 
    train_loss = train_loop(training_loader,model,optimizer,hyperparameter,device)
    validation_loss = validation_loop(validation_loader,model,hyperparameter,device)
    
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Validation Loss: {validation_loss:.3f}")

   
torch.save(model.state_dict(), f"{MODEL_DIR}/transformer_model.pth") # Store the whole model

torch.save(model.encoder.state_dict(), f"{MODEL_DIR}/transformer_encoder.pth") # Store the encoder only 

with open(f"{MODEL_DIR}/architecture_parameters.json","w") as f:
    json.dump(architecture_parameter,f) # Store architecture parameters 