import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim,dim,bias=False)
        self.proj_2 =  nn.Linear(dim,dim,bias=False)        
        self.gelu = nn.GELU()
       
             	   
    def forward(self, x):
       
        x = self.proj_1(x)
        x = self.gelu(x)          
        x = self.proj_2(x)
        
                          
        return x
        

class Context_PReLUBlock(nn.Module):

    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.context_prelu = nn.PReLU(d_model * num_tokens)
        self.token_norm = nn.LayerNorm(d_model)
        self.local_mapping = MLP(d_model)
                        
        
    def forward(self, x):
                  
       
        
        residual = x
        
        x = self.token_norm(x)
        
        dim0 = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        context = x.reshape([dim0,dim1*dim2])
        
                       
        readout = self.context_prelu(context)        
        
        x = readout.reshape([dim0,dim1,dim2])
            
        x = x + residual
            
        residual = x

        x = self.local_mapping(x)
                                          
        out = x + residual
        
        
        return out



class Context_PReLU(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[Context_PReLUBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








