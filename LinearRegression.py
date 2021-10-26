import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#Data Generation
input_data = np.random.randint(1,10,(1000,1))
output_data = 2*np.array(input_data)

#Data Processing
X = torch.from_numpy(input_data.astype(np.float32))
Y = torch.from_numpy(output_data.astype(np.float32))
y = Y.view(Y.shape[0], 1)

#Model Definition
sample_size, input_size = X.shape
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer Definition
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Training
num_epochs = 100
alpha = 0
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    # Plotting
    predicted = model(X).detach().numpy()

    plt.plot(X, Y, 'ro')
    plt.plot(X, predicted, 'b', alpha=alpha)
    alpha += 0.01
#Show final graph
plt.show()