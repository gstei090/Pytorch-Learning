import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.sig = nn.Sigmoid().to(device)
        self.l1 = nn.Linear(input_size, hidden_layer_size).to(device)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)  
        self.l3 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
        self.l4 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
        self.l5 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
        self.l6 = nn.Linear(hidden_layer_size, output_size).to(device)


    def forward(self, x):
        out = self.l1(x).to(device)
        out = self.sig(out).to(device)
        out = self.l2(out).to(device)
        out = self.sig(out).to(device)
        out = self.l3(out).to(device)
        out = self.sig(out).to(device)
        out = self.l4(out).to(device)
        out = self.sig(out).to(device)
        out = self.l5(out).to(device)
        out = self.sig(out).to(device)
        out = self.l6(out).to(device)

        return out

#Data Generation
def polynomial_func(x):
   return x**3 - 10*x**2 + 4*x - 3

input_data = np.random.randint(1,10,(10000,1))
output_data = np.array(input_data)
for i in range(len(output_data)):
    output_data[i][0]=polynomial_func(output_data[i][0])

#Data Processing
x_cpu = torch.from_numpy(input_data.astype(np.float32))
X = x_cpu.to(device)
y_cpu = torch.from_numpy(output_data.astype(np.float32))
Y = y_cpu.to(device)
y = Y.view(Y.shape[0], 1)

#Model Definition
sample_size, input_size = X.shape
output_size = 1
hidden_layer_size = 50
model = NeuralNet(input_size, output_size, hidden_layer_size)

# Loss and Optimizer Definition
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Training
num_epochs = 10000
alpha = 0
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X).to(device)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

        # Plotting the predictions of the model during the learning process
        predicted = model(X).detach().cpu().numpy()
        plt.plot(x_cpu, predicted, 'bo', alpha=alpha)
        alpha += 0.005

#This is to show the final prediction made by the model (Blue stars)
# and the initial output data to be compared to (Red squares)
predicted = model(X).detach().cpu().numpy()
plt.plot(x_cpu, y_cpu, 'rs')
plt.plot(x_cpu, predicted, 'b*')

#Show the final graph
plt.show()

# TODO
# 1. met plus de points et fait un split training and testing. genre 80%-20%. 
# 2. ajoute du bruit poluynomial aux inputs et outputs
# 3. Essaie ADAM au lieu SGD
# 4. rapporte tes resultats sur le test.
# 5. Use CUDA as the device