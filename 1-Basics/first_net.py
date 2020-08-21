import os

import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #                 input   output     -> output muss mit input der nächsten Schicht übereinstimmen
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]   # batch
        num = 1
        for i in size:
            num *= i
        return num


# run on gpu -> .cuda

#net = Network() 
#print(net)
#net = net.cuda()

# Load
if os.path.isfile('my_net.pt'):
    net = torch.load('my_net.pt')

for i in range(100):
    x = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    input = input.cuda()

    out = net(input)

    x = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    target = target.cuda()
    criterion = nn.MSELoss()
    loss = criterion(out, target) # compare target and output  -> lossfunction
    print(loss)

    net.zero_grad()
    loss.backward()
    optimizer = SGD(net.parameters(), lr=0.1)
    optimizer.step()

# Save
torch.save(net, 'my_net.pt')


        """VSCode Extension : pytorch snippets
        USAGE: pyt.... use for example pytorch:module
        """