import torch.nn as nn
from model.model import TransformerEncoder


class SingleRegressionModel(nn.Module):
    
    def __init__(self,hidden_dim, output_dim,encoder:TransformerEncoder):
        super().__init__()

        self.encoder = encoder
        input_dim = encoder.layers[0].embedding_dim

        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.SiLU()
        self.linear1 = nn.Linear(input_dim,output_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)

        #register encoder parameters 


        
        
    def forward(self,x,padding_mask=None):
        x = self.encoder(x,padding_mask)
        #x = self.norm(x)
        x = self.activation(x)
        x = self.linear1(x)
        #x= self.norm2(x)
        #x = self.activation(x)
        #x = self.linear2(x)

        return x
    

class MultiTaskRegressionModel(nn.Module):
    pass