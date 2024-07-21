# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Check stopwords
print(stopwords.words('english'))

# Load and prepare the data
df = pd.read_csv('train.csv')
df = df.fillna('')
df['content'] = df['author'] + ' ' + df['title']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['content'] = df['content'].apply(stemming)

x = df['content'].values
y = df['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=7)

# Build the neural network
model = Sequential()
model.add(Dense(128, input_dim=xtrain.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(xtrain.toarray(), ytrain, epochs=10, batch_size=64, validation_data=(xtest.toarray(), ytest))

# Evaluate the model
ypred = (model.predict(xtest.toarray()) > 0.5).astype("int32")
ypred = ypred.flatten()
acs = accuracy_score(ytest, ypred)
print("Accuracy:", acs)

# Save the model
model.save('nnclassifier_model369.h5')

# Validation Code
def preprocess_text(text):
    # Perform the same preprocessing as for training
    stemmed_text = stemming(text)
    vectorized_text = vectorizer.transform([stemmed_text])
    return vectorized_text

def predict_text(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Load the trained model
    model = load_model('nn_model369.h5')
    # Make a prediction
    prediction = model.predict(preprocessed_text.toarray())
    # Get the class with the highest probability
    prediction_class = (prediction > 0.5).astype("int32")[0][0]
    print(f"The text is classified as: {'Class 1' if prediction_class == 1 else 'Class 0'}")

# Example usage
example_text = "Sample author Sample title"
predict_text(example_text)
