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

def polynomial_func(x):
   return x**3 - 10*x**2 + 4*x - 3

#Data Generation
num_data_points = 20000
raw_input = np.random.randint(1,10,(num_data_points,1))
raw_output = np.array(raw_input)
for i in range(len(raw_output)):
    raw_output[i][0]=polynomial_func(raw_output[i][0])

    #Generate some noise in the outputs
    noise = np.random.randint(0,5)
    if noise == 0:
        raw_output[i][0] += np.random.randint(1,10)

input_data = raw_input[:int(num_data_points*0.8)]
input_test_data = raw_input[int(num_data_points*0.8):]

output_data = raw_output[:int(num_data_points*0.8)]
output_test_data = raw_output[int(num_data_points*0.8):]

#Data Processing
#Generating the tensors for the input data and input test data (X and X_test)
x_cpu = torch.from_numpy(input_data.astype(np.float32))
X = x_cpu.to(device)
x_test_cpu = torch.from_numpy(input_test_data.astype(np.float32))
X_test = x_test_cpu.to(device)

#Generating the output data
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

#Testing the accuray of the model
with torch.no_grad():
    correct_predictions = 0
    num_samples = 0
    output = model(X_test).to(device)
    for i in range(output.size()[0]):
        output_difference = output_test_data[i][0] - output[i][0]
        if abs(output_difference) < 1:
            correct_predictions += 1
        num_samples += 1
    accuracy = 100 * (correct_predictions/num_samples)
    print(f"accuracy of model = {correct_predictions}/{num_samples} = {accuracy}%")

#This is to show the final prediction made by the model (Blue stars)
# and the initial output data to be compared to (Red squares)
predicted = model(X).detach().cpu().numpy()
plt.plot(x_cpu, y_cpu, 'rs')
plt.plot(x_cpu, predicted, 'b*')

#Show the final graph
plt.show()