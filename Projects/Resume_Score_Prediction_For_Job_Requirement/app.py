import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import docx2txt #for docx files

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) #for lemmatizer
nltk.download('stopwords', quiet=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')]) #remove stop words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # lemmatize
    return text


st.title("Resume Scoring Application")

job_description = st.text_area("Enter Job Description:", height=200)

uploaded_files = st.file_uploader("Choose Resume Files", type=["txt", "docx"], accept_multiple_files=True)

if job_description and uploaded_files:
    try:
        processed_job_description = preprocess_text(job_description)
        vectorizer = TfidfVectorizer()
        vectorizer.fit([processed_job_description])

        resume_texts = []
        for file in uploaded_files:
            if file.name.endswith(".txt"):
                resume_text = file.read().decode("utf-8")
            elif file.name.endswith(".docx"):
                resume_text = docx2txt.process(file.getvalue())
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue  # Skip to the next file

            processed_resume = preprocess_text(resume_text)
            resume_texts.append(processed_resume)

        if not resume_texts: #handle case where no valid files were processed
            st.warning("No valid resume files processed. Please upload .txt or .docx files.")
            st.stop()

        job_description_vector = vectorizer.transform([processed_job_description])
        resume_vectors = vectorizer.transform(resume_texts)

        similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()

        for i, score in enumerate(similarities):
            st.write(f"Resume {i+1} Score: {score:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
