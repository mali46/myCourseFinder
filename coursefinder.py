import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import streamlit as st 

nltk.download('punkt')  
from nltk.corpus import stopwords

try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    text = ' '.join(filtered_tokens)
    return text

df = pd.read_csv('yes.csv', encoding='utf-8', on_bad_lines='skip', delimiter=";")

categories = {
    "1": ["basic", "easy", "intro", "relaxed", "chill"],
    "2": ["medium", "intermediate", "not too hard"],
    "3": ["advanced", "difficult", "hard"],
}

df['Course Name'] = df['Course Name'].astype(str)
df['Course Description'] = df['Course Description'].astype(str).apply(preprocess)
df['Summary'] = df['Summary'].astype(str).apply(preprocess)
df['Subject'] = df['Subject'].astype(str).apply(preprocess)

user_input = st.text_input("What course are you looking for? ")
user_input_preprocessed = preprocess(user_input)

category = None
for cat, keywords in categories.items():
    if any(keyword in user_input_preprocessed for keyword in keywords):
            category = cat
            break

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
preferred_days = [day for day in days if day.lower() in user_input_preprocessed]

time_of_day = None
for time in ["morning", "afternoon", "evening"]:
    if time in user_input_preprocessed:
        time_of_day = time.capitalize()
        break

# Apply the filters
if category is not None:
    df = df[df['category'] == category]
if preferred_days:
    df = df[df[preferred_days].sum(axis=1) > 0]
if time_of_day is not None:
    df = df[df['TimeOfDay'].str.lower() == time_of_day]

vectorizer_name = CountVectorizer()
vectorizer_desc = CountVectorizer()
vectorizer_summary = CountVectorizer()
vectorizer_subject = CountVectorizer()

# Set the weights
weight_name = 0.2
weight_desc = 0.20
weight_summary = 0.10
weight_subject = 0.5

# Calculate the weighted similarities for the filtered courses
count_matrix_name = vectorizer_name.fit_transform(df['Course Name'])
count_matrix_desc = vectorizer_desc.fit_transform(df['Course Description'])
count_matrix_summary = vectorizer_summary.fit_transform(df['Summary'])
count_matrix_subject = vectorizer_subject.fit_transform(df['Subject'])

similarity_name = weight_name * (count_matrix_name * vectorizer_name.transform([user_input_preprocessed]).T)
similarity_desc = weight_desc * (count_matrix_desc * vectorizer_desc.transform([user_input_preprocessed]).T)
similarity_summary = weight_summary * (count_matrix_summary * vectorizer_summary.transform([user_input_preprocessed]).T)
similarity_subject = weight_subject * (count_matrix_subject * vectorizer_subject.transform([user_input_preprocessed]).T)

# Sum the weighted similarities
total_similarity = similarity_name + similarity_desc + similarity_summary + similarity_subject

# Get the top matches based on the total similarity
top_matches = total_similarity.toarray().flatten().argsort()[-8:]

filtered_matches = []
for match in top_matches[::-1]: 
    filtered_matches.append(match)
    if len(filtered_matches) == 8:
        break

for i, match in enumerate(filtered_matches, start=1):
    st.write(f"{i}. Course Code: {df.iloc[match]['Course Code']}, Course Name: {df.iloc[match]['Course Name']}\n")
