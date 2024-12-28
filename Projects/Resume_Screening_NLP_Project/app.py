import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt', quiet=True) #Download Punkt Sentence Tokenizer
nltk.download('stopwords', quiet=True) #Download Stopwords


# Load the dataset.  Make sure "UpdatedResumeDataSet.csv" is in the same directory.
try:
    df = pd.read_csv("UpdatedResumeDataSet.csv")
except FileNotFoundError:
    st.error("Error: 'UpdatedResumeDataSet.csv' not found. Please make sure the dataset is in the same directory as your Streamlit app.")
    st.stop()


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub('[^A-Za-z0-9\s]+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['Resume'] = df['Resume'].apply(lambda x: preprocess_text(x))

# Label Encoding
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf.fit(df['Resume'])
vector = tfidf.transform(df['Resume'])

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(vector, df['Category'], test_size=0.2, random_state=42)

# Model Training (Choose one model -  uncomment the desired model)
#KNN
#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)

#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#SVM
#svm = SVC()
#svm.fit(X_train, y_train)

#Logistic Regression
#lr = LogisticRegression()
#lr.fit(X_train, y_train)


job_titles = ['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
              'Mechanical Engineer', 'Sales', 'Health and fitness',
              'Civil Engineer', 'Java Developer', 'Business Analyst',
              'SAP Developer', 'Automation Testing', 'Electrical Engineering',
              'Operations Manager', 'Python Developer', 'DevOps Engineer',
              'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
              'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
indices = [6, 12, 0, 1, 24, 16, 22, 14, 5, 15, 4, 21, 2, 11, 18, 20, 8, 17, 19, 7, 13, 10, 9, 3, 23]

index_to_job_title = {index: job_title for index, job_title in zip(indices, job_titles)}


# Streamlit app
st.title("Resume Category Prediction")

uploaded_file = st.file_uploader("Choose a resume file", type=["txt"])

if uploaded_file is not None:
    try:
        resume_text = uploaded_file.read().decode("utf-8")
        clean_resume = preprocess_text(resume_text)
        input_features = tfidf.transform([clean_resume])

        #Prediction (Choose the trained model - uncomment the relevant line)
        #pred_resume_id = knn.predict(input_features)[0]
        pred_resume_id = rf.predict(input_features)[0]
        #pred_resume_id = svm.predict(input_features)[0]
        #pred_resume_id = lr.predict(input_features)[0]

        predicted_job_title = index_to_job_title.get(pred_resume_id, "Job title not found")
        st.write(f"Predicted Job Category: {predicted_job_title}")

    except Exception as e:
        st.error(f"An error occurred: {e}")