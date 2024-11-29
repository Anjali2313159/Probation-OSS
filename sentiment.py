# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load the dataset
# For this example, we will use the IMDB dataset from Kaggle or any other source.
# Assuming the dataset is in CSV format with columns 'review' and 'sentiment'
data = pd.read_csv('movie_reviews.csv')  # Replace with your dataset path

# Step 2: Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Visualizing the distribution of sentiments
sns.countplot(x='sentiment', data=data)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.show()

# Step 3: Data Preprocessing
# Text cleaning function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english') and word.isalpha()]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Step 4: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 6: Initialize the Naive Bayes Classifier
model = MultinomialNB()

# Step 7: Hyperparameter Tuning using Grid Search
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters from Grid Search: {grid_search.best_params_}")

# Step 8: Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 9: Make predictions
predictions = best_model.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Step 11: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 12: Classifying new reviews
def classify_review(review):
    cleaned_review = preprocess_text(review)
    features = vectorizer.transform([cleaned_review]).toarray()
    sentiment = best_model.predict(features)
    return "Positive" if sentiment[0] == 1 else "Negative"

# Example usage
new_review = "This movie was fantastic! I loved every moment of it."
print("Sentiment of the new review:", classify_review(new_review))

new_review = "This movie was terrible and a waste of time."
print("Sentiment of the new review:", classify_review(new_review))