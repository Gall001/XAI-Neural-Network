import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# Load the dataset
dataset_path = 'IMDB Dataset.csv'
df = pd.read_csv(dataset_path)

# Split the data into training and testing sets
X = df['review'].values
Y = (df['sentiment'] == 'positive').astype(int).values  # Convert sentiment to binary labels (0 for negative, 1 for positive)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tokenize the text data using a CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # Limit vocabulary to the top 5000 words
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 64 hidden units
        self.fc2 = nn.Linear(64, 1)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

# Initialize the model
input_dim = X_train_vec.shape[1]
model = SimpleNN(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Convert the sparse matrix to a dense tensor
    X_train_vec_dense = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
    
    # Forward pass
    outputs = model(X_train_vec_dense)
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluation on the test set
model.eval()
with torch.no_grad():
    # Convert the sparse matrix to a dense tensor
    X_test_vec_dense = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
    test_outputs = model(X_test_vec_dense)
    predicted = (test_outputs >= 0.5).squeeze().cpu().numpy()

accuracy = accuracy_score(y_test, predicted)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
