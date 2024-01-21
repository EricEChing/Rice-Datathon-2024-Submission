import torch

class ChevronModel(torch.nn.Module):

    def __init__(self):
        super(ChevronModel, self).__init__()
        self.linear = torch.nn.Linear(7, 1)  
        self.droupout = torch.nn.AlphaDropout()
        self.activation = torch.nn.LeakyReLU()


    def forward(self, x):
        x = self.linear(x)
        x = self.droupout(x)
        x = self.activation(x)
        return x
