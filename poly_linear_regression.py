import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, output_size) 
        self.l2 = nn.Linear(input_size, output_size)  
        self.l3 = nn.Linear(input_size, output_size)  
        self.l4 = nn.Linear(input_size, output_size)  
        self.l5 = nn.Linear(input_size, output_size)  
        self.l6 = nn.Linear(input_size, output_size)  


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        # no activation and no softmax at the end
        return out

#Data Generation
def polynomial_func(x):
    return x**3 - 10*x**2 + 4*x - 3

input_data = np.random.randint(1,10,(100,1))
output_data = np.array(input_data)
for i in range(len(output_data)):
    output_data[i][0]=polynomial_func(output_data[i][0])

#Data Processing
X = torch.from_numpy(input_data.astype(np.float32))
Y = torch.from_numpy(output_data.astype(np.float32))
y = Y.view(Y.shape[0], 1)

#Model Definition
sample_size, input_size = X.shape
output_size = 1
model = NeuralNet(input_size, output_size)

# Loss and Optimizer Definition
learning_rate = 0.0001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Training
num_epochs = 1000
alpha = 0
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    # Plotting
    predicted = model(X).detach().numpy()

    plt.plot(X, Y, 'ro')
    plt.plot(X, predicted, 'b', alpha=alpha)
    alpha += 0.0001

#Show final graph
plt.show()


