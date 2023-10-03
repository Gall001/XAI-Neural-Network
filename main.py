import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import random

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

num_decimal_places = 2  # Specify the number of decimal places you want to display

# Randomly select 10 reviews from your dataset (adjust the number as needed)
random_reviews = random.sample(df['review'].tolist(), 5000)

# Initialize variables to store word importance scores for positive and negative reviews
positive_scores = {}
negative_scores = {}

def predict_batch(texts):
    # Convert the text data to the required format (e.g., using the CountVectorizer)
    text_data = vectorizer.transform(texts)
    text_data = torch.tensor(text_data.toarray(), dtype=torch.float32)

    # Perform model inference
    model.eval()
    with torch.no_grad():
        outputs = model(text_data)

    # Convert the model outputs to a NumPy array
    return np.hstack((1 - outputs, outputs))  # Include probabilities for both classes

# Define a custom tokenizer function
def custom_tokenizer(text):
    return text.split()  # Split text on whitespace to keep complete words

# Function to generate LIME explanations for a given text
def generate_lime_explanations(text, predict_fn, num_features=5000, labels=[0, 1]):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"], split_expression=custom_tokenizer)
    explanation = explainer.explain_instance(text, predict_fn, num_features=num_features, labels=labels)
    return explanation

def print_top_words(words_scores, label):
    total_score = sum(score for _, score in words_scores)  # Calculate total score for normalization
    print(f"Top words suggesting {'Positive' if label == 1 else 'Negative'} Reviews:")
    for word, score in words_scores:
        normalized_score = (score / total_score) * 100  # Normalize the score to a percentage
        print(f"{word}: {normalized_score:.2f}%")

############################################ SIMPLE NETWORK ############################################

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

# Generate LIME explanations for each random review
for review in random_reviews:
    explanation = generate_lime_explanations(review, predict_batch)
    
    # Aggregate word importance scores for positive and negative labels
    for word, score in explanation.as_list(label=1):
        positive_scores[word] = positive_scores.get(word, 0) + score
    for word, score in explanation.as_list(label=0):
        negative_scores[word] = negative_scores.get(word, 0) + score

# Sort and print the top words for positive and negative reviews
top_positive_words = sorted(positive_scores.items(), key=lambda x: x[1], reverse=True)
top_negative_words = sorted(negative_scores.items(), key=lambda x: x[1], reverse=True)

# Sort and print the top words for positive and negative reviews as percentages

# Sort and print the top words for positive and negative reviews
print_top_words(top_positive_words[:10], label=1)
print("\n")
print_top_words(top_negative_words[:10], label=0)


############################################ COMPLEX NETWORK ############################################

# Define a neural network model with an additional hidden layer
class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 64 hidden units for the first hidden layer
        self.fc2 = nn.Linear(64, 32)  # 32 hidden units for the second hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Applying ReLU activation for the second hidden layer
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x

# Initialize the complex model
complex_model = ComplexNN(input_dim)

# Define loss function and optimizer for the complex model
complex_optimizer = optim.Adam(complex_model.parameters(), lr=0.001)

# Training loop for the complex model
for epoch in range(num_epochs):
    complex_model.train()
    complex_optimizer.zero_grad()

    X_train_vec_dense = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
    outputs = complex_model(X_train_vec_dense)
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))

    loss.backward()
    complex_optimizer.step()

    print(f'Complex Model - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluation on the test set for the complex model
complex_model.eval()
with torch.no_grad():
    X_test_vec_dense = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
    complex_test_outputs = complex_model(X_test_vec_dense)
    complex_predicted = (complex_test_outputs >= 0.5).squeeze().cpu().numpy()

complex_accuracy = accuracy_score(y_test, complex_predicted)
print(f'Complex Model Test Accuracy: {complex_accuracy * 100:.2f}%')

# Generate LIME explanations for the complex model
positive_scores_complex = {}
negative_scores_complex = {}

for review in random_reviews:
    explanation = generate_lime_explanations(review, predict_batch)

    for word, score in explanation.as_list(label=1):
        positive_scores_complex[word] = positive_scores_complex.get(word, 0) + score
    for word, score in explanation.as_list(label=0):
        negative_scores_complex[word] = negative_scores_complex.get(word, 0) + score

top_positive_words_complex = sorted(positive_scores_complex.items(), key=lambda x: x[1], reverse=True)
top_negative_words_complex = sorted(negative_scores_complex.items(), key=lambda x: x[1], reverse=True)

# Sort and print the top words for positive and negative reviews
print_top_words(top_positive_words_complex[:10], label=1)
print("\n")
print_top_words(top_negative_words_complex[:10], label=0)
