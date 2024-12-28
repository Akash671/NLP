"""
author : @akahs
"""

print("reading libraries...")

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import string
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from spellchecker import SpellChecker
import nltk
nltk.download('punkt_tab')
print("done")

print("loading data...")
df = pd.read_csv("UpdatedResumeDataSet.csv")
print("done")

print("data preprocessing is ongoing..")
# text clean preprocessing step-------------
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove special characters
    text = re.sub('[^A-Za-z0-9\s]+', '', text)

    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Remove newline characters, carriage return characters, and other special characters
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    return text

df['Resume'] = df['Resume'].apply(lambda x: preprocess_text(x))
print("data preprocessing done")

print("Label Encoding is ongoing...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
print("done")

print("vectorization is ongoing...")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

tfidf.fit(df['Resume'])
vector = tfidf.transform(df['Resume'])
print("done")

print("data splitting is ongoing...")
# split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vector, df['Category'], test_size=0.2, random_state=42)
print("done")

print("model training is ongoing...")
# Initialize classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression()
}

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {clf_name}: {accuracy}')

print("done")

print("@@@----------------prediction face--------------------@@@")

while True:
    file_path = input("Enter the path to the resume text file: ")

    # Read resume text from a file
    def read_resume_from_file(file_path):
        with open(file_path, 'r') as file:
            resume_text = file.read()
        return resume_text

    my_resume = read_resume_from_file(file_path)

    clean_resume = preprocess_text(my_resume)
    input_features = tfidf.transform([clean_resume])
    pred_resume_id = clf.predict(input_features)[0]

    job_titles = ['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
                  'Mechanical Engineer', 'Sales', 'Health and fitness',
                  'Civil Engineer', 'Java Developer', 'Business Analyst',
                  'SAP Developer', 'Automation Testing', 'Electrical Engineering',
                  'Operations Manager', 'Python Developer', 'DevOps Engineer',
                  'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
                  'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
    indices = [6, 12, 0, 1, 24, 16, 22, 14, 5, 15, 4, 21, 2, 11, 18, 20, 8, 17, 19, 7, 13, 10, 9, 3, 23]

    # Create a dictionary to map indices to job titles
    index_to_job_title = {index: job_title for index, job_title in zip(indices, job_titles)}

    # Map the predicted index to job title
    job_title = index_to_job_title.get(pred_resume_id, "Job title not found")
    print(f"The job title at index {pred_resume_id} is: {job_title}")

    key = input("Want to continue? (yes/no): ")
    if key.lower() != "yes":
        break