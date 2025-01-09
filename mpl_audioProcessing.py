import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

class AudioMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AudioMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Sample learning algorithm implementation
def train_model(model, data_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            # Flatten the input data
            inputs = inputs.view(inputs.size(0), -1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage
input_size = 128  # Example input size for audio features
hidden_size = 64
output_size = 10  # Example number of output classes

model = AudioMLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data loader (dummy data)
data_loader = [(torch.randn(32, input_size), torch.randint(0, output_size, (32,))) for _ in range(100)]

# Train the model
train_model(model, data_loader, criterion, optimizer, num_epochs=5)
