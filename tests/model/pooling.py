import torch
from threedscriptors.model.pooling import AttnPool, ChiralAttnPool



def test_pooling_preserves_pseudoscalar_information():

    N = 2
    d_even = 100
    d_odd = 100

    pool = AttnPool(d_in = d_even+d_odd)


    X_even = torch.randn(1, 2, d_even)
    X_odd  = torch.randn(1, 1, d_odd)
    X_odd_2 =  -1*  X_odd 

    X_odd = torch.cat([X_odd, X_odd_2], dim = 1)
    X      = torch.cat([X_even, X_odd], -1)

    X_mirror = torch.cat([ X_even,
                        -X_odd ], -1)         # reflect pseudoscalars only

    g1 = pool(X)          # (1,D)
    g2 = pool(X_mirror)   # (1,D)

    print("NonChiral")
    print(torch.linalg.norm(g1 - g2)) 


    parity = torch.torch.BoolTensor([False]*d_even + [True]*d_odd)

    pool = ChiralAttnPool(d_in = d_even+d_odd, parity= parity)

    print("Chiral")
    g1_chir = pool(X)
    g2_chir = pool(X_mirror)   # (1,D)
    print(torch.linalg.norm(g1_chir - g2_chir).detach().item()) 
    
