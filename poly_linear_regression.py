import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

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

def polynomial_func(x):
   return x**3 - 10*x**2 + 4*x - 0.5

def random_float(low, high):
    return random.random()*(high-low) + low

#Data Generation
num_data_points = 20000

input_data = torch.rand(num_data_points,1).to(device)
output_data = torch.empty(num_data_points,1).to(device)
for i in range(len(output_data)):
    output_data[i][0]=polynomial_func(input_data[i][0])

    #Generate some noise in the outputs
    noise = np.random.randint(0,5)
    if noise == 0:
        output_data[i][0] += random_float(0,1)

#Data Processing
X = input_data[:int(num_data_points*0.8)].to(device)
X_test = input_data[int(num_data_points*0.8):].to(device)

Y = output_data[:int(num_data_points*0.8)].to(device)
Y_test = output_data[int(num_data_points*0.8):].to(device)
y = Y.view(Y.shape[0], 1).to(device)
y_test = Y_test.view(Y_test.shape[0], 1).to(device)

#Model Definition
sample_size, input_size = X.shape
output_size = 1
hidden_layer_size = 50
model = NeuralNet(input_size, output_size, hidden_layer_size)

# Loss and Optimizer Definition
learning_rate = 0.1
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
        predicted = model(X).detach().cpu()
        a = X.view(-1).detach().cpu().numpy()
        b = predicted.view(-1).numpy()
        coefficients = np.polyfit(a, b, 3)
        poly = np.poly1d(coefficients)
        new_x = np.linspace(0, 10)
        new_y = poly(new_x)
        plt.plot(new_x, new_y, 'b', alpha=alpha)
        alpha += 0.1

#Testing the accuray of the model
with torch.no_grad():
    y_test_predicted = model(X_test).to(device)
    loss = criterion(y_test_predicted, y_test)
    print(f"Mean Squared Loss of the model using test data = {loss}")

#This is to show the final prediction made by the model (Blue stars)
# and the initial output data to be compared to (Red squares)
predicted = model(X).detach().cpu().numpy()
a = X.view(-1).detach().cpu().numpy()
b = Y.view(-1).detach().cpu().numpy()
coefficients = np.polyfit(a, b, 3)
poly = np.poly1d(coefficients)
new_x = np.linspace(0, 10)
new_y = poly(new_x)
plt.plot(new_x, new_y, 'r')

#Show the final graph
plt.show()