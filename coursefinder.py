import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import streamlit as st
import numpy as np

nltk.download('punkt')  
from nltk.corpus import stopwords

try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

st.write("MyCourseFinder")
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    text = ' '.join(filtered_tokens)
    return text

df = pd.read_csv('yes3.csv', encoding='utf-8', on_bad_lines='skip', delimiter=";")

categories = {
    "1": ["basic", "easy", "intro", "relaxed", "chill", "fun"],
    "2": ["medium", "intermediate", "not too hard"],
    "3": ["advanced", "difficult", "hard"],
}

df['Subject'] = df['Subject'].astype(str)
df['Course Name'] = df['Course Name'].astype(str)
df['Course Description'] = df['Course Description'].astype(str)
df['Summary'] = df['Summary'].astype(str)

user_input = st.text_input("What course are you looking for? ")
user_input_preprocessed = preprocess(user_input)

# Extract filters
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
preferred_days = [day for day in days if day.lower() in user_input_preprocessed]

category = None
for cat, keywords in categories.items():
    if any(keyword in user_input for keyword in keywords):
        category = cat
        break

time_of_day = None
for time in ["morning", "afternoon", "evening"]:
    if time in user_input_preprocessed:
        time_of_day = time.capitalize()
        break

# Apply filters
df_filtered = df.copy()
if category:
    df_filtered = df_filtered[df_filtered['category'] == int(category)]
if preferred_days:
    df_filtered = df_filtered[df_filtered[preferred_days].sum(axis=1) > 0]
if time_of_day:
    df_filtered = df_filtered[df_filtered['TimeOfDay'].str.lower() == time_of_day.lower()]

if df_filtered.empty:
    st.write("No courses found that match your criteria.")
else:
    # Vectorize
    vectorizer_name = CountVectorizer()
    vectorizer_desc = CountVectorizer()
    vectorizer_summary = CountVectorizer()
    vectorizer_subject = CountVectorizer()

    count_matrix_name = vectorizer_name.fit_transform(df_filtered['Course Name'])
    count_matrix_desc = vectorizer_desc.fit_transform(df_filtered['Course Description'])
    count_matrix_summary = vectorizer_summary.fit_transform(df_filtered['Summary'])
    count_matrix_subject = vectorizer_subject.fit_transform(df_filtered['Subject'])

    # Set the weights
    weight_name = 0.25
    weight_desc = 0.2
    weight_summary = 0.1
    weight_subject = 0.45

    # Calculate the weighted similarities
    similarity_name = weight_name * (count_matrix_name * vectorizer_name.transform([user_input_preprocessed]).T)
    similarity_desc = weight_desc * (count_matrix_desc * vectorizer_desc.transform([user_input_preprocessed]).T)
    similarity_summary = weight_summary * (count_matrix_summary * vectorizer_summary.transform([user_input_preprocessed]).T)
    similarity_subject = weight_subject * (count_matrix_subject * vectorizer_subject.transform([user_input_preprocessed]).T)

    # Sum the weighted similarities
    total_similarity = similarity_name + similarity_desc + similarity_summary + similarity_subject

    # Get the top matches based on the total similarity
    top_matches = total_similarity.toarray().flatten().argsort()[-8:]

    for i, match in enumerate(top_matches[::-1], start=1):
        st.write(f"{i}. Course Code: {df_filtered.iloc[match]['Course Code']}, Course Name: {df_filtered.iloc[match]['Course Name']}\n")
