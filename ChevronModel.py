import torch

class ChevronModel(torch.nn.Module):

    def __init__(self):
        super(ChevronModel, self).__init__()
        self.linear1 = torch.nn.Linear(7, 4)
        self.dropout = torch.nn.Dropout(p=0.75)    
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 1) 
        self.activation2 = torch.nn.ReLU()
        '''
        self.linear2 = torch.nn.Linear(16, 8)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(8, 4)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(4, 1)
        self.activation4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(2, 1)
        self.activation5 = torch.nn.ReLU()
        '''

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation2(x)
        '''
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        x = self.activation5(x)
        '''
        return x