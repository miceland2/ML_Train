"""
Implementation of a MLP with two hidden units to learn the XOR function
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0], [1], [1], [0]])

class XORmodel(nn.Module):
    
    def __init__(self):
        super(XORmodel, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
        self.Mish = nn.Mish()
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.15)
        
    def forward(self, x):
        x = self.Mish(self.linear1(x))
        x = self.linear2(x)
        return x
    
model = XORmodel()

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.015
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_epochs = 1000

losses_h = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    out = model(X)
    loss = criterion(out, y.float())
    
    loss.backward()
    optimizer.step()
    
    if ((epoch + 1) % 1 == 0):
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
        losses_h.append(loss.item())
        
### plot the decision boundary

def decision_boundary(x, y):
    color_map = plt.get_cmap('bwr')

    xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
    ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    x_span = np.linspace(xmin, xmax, num_epochs)
    y_span = np.linspace(ymin, ymax, num_epochs)
    
    xx, yy = np.meshgrid(x_span, y_span)

    model.eval()
    labels_predicted = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))

    labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    z = np.array(labels_predicted).reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)

    train_labels_predicted = model(x)
    ax.scatter(x[:, 0], x[:, 1], c=y.reshape(y.size()[0]), cmap=color_map, lw=0)
    plt.title('Decision Boundary for Network with a Hidden Layer')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
    return fig, ax

decision_boundary(X, y)