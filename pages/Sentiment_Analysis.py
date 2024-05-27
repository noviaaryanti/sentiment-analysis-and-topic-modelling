import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load stopwords
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
# Load additional stopwords from Excel file
more_stopwords = pd.read_excel('data/stopwords.xlsx')
stopwords_from_excel = set(more_stopwords['stopword'].tolist())
stopwords.extend(stopwords_from_excel)
# Exception words
exceptions = set(['tidak', 'belum', 'kurang', 'jangan'])

# Initialize stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

@st.cache_resource()
def load_model():
    model_path = 'models/svm_model_poly.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model atau vectorizer tidak ditemukan. Pastikan file `svm_model_poly.pkl` dan `tfidf_vectorizer.pkl` tersedia.")
        return None, None
    
    with open(model_path, 'rb') as model_file, open(vectorizer_path, 'rb') as vectorizer_file:
        svm_model = pickle.load(model_file)
        tfidf_vectorizer = pickle.load(vectorizer_file)
    return svm_model, tfidf_vectorizer

# Preprocessing functions
def remove_mention_hashtag(text):
    text = re.sub(r'@[A-Za-z0-9_.-]{1,50}', '', text) # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)        # Remove hashtags
    return text

def remove_url(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    return text

def remove_emoticons(text):
    emoj = pd.read_csv('data/emojis.csv')
    emojis = list(emoj['emoji'])
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in emojis) + u')')
    text = re.sub(emoticon_pattern," ", text)
    return text

def remove_special_character(text):
    text = re.sub(r'<.*?>|&[a-z]+|href', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]',' ', text)              # Remove punctuations
    text = re.sub(r'[^0-9A-Za-z]+', ' ', text)       # Remove non-alphanumeric
    text = re.sub(r'[-+]?[0-9]+', ' ', text)         # Remove numbers
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)        # Remove single characters
    text = re.sub(r'^\s+', '', text)                 # Remove leading spaces
    text = re.sub(r' +', ' ', text)                  # Remove extra spaces
    return text

def case_folding(text):
    return text.lower()

def normalize(text):
    key_norm = pd.read_excel('data/key_norm.xlsx')
    text = ' '.join([key_norm[key_norm['tidak_baku'] == word]['baku'].values[0] if (key_norm['tidak_baku'] == word).any() else word for word in text.split()])
    return text

def remove_stopwords(text):
  clean_words = []
  text = text.split()
  for word in text:
      if word not in stopwords or word in exceptions:
          clean_words.append(word)
  return " ".join(clean_words)

def stemming(text):
  text = stemmer.stem(text)
  return text

def gabung_negasi(text):
    negasi_list = ['tidak', 'belum', 'kurang', 'jangan']
    for negasi in negasi_list:
        pattern = r'\b' + negasi + r'\s+(\w+)'
        text = re.sub(pattern, negasi + r'\1', text)
    return text

# Combine preprocessing functions
def preprocess_data(df, column_name):
    df[column_name] = df[column_name].apply(remove_mention_hashtag)
    df[column_name] = df[column_name].apply(remove_url)
    df[column_name] = df[column_name].apply(remove_emoticons)
    df[column_name] = df[column_name].apply(remove_special_character)
    df[column_name] = df[column_name].apply(case_folding)
    df[column_name] = df[column_name].apply(normalize)
    df[column_name] = df[column_name].apply(stemming)
    df[column_name] = df[column_name].apply(gabung_negasi)
    df[column_name] = df[column_name].apply(remove_stopwords)
    return df[column_name].values

def main():
    st.title("Sentimen Analysis")
    st.write("Di halaman ini, Anda dapat melakukan analisis sentimen pada data kendaraan listrik.")

    st.write("### Input Data")
    uploaded_file = st.file_uploader("Upload file CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah! Berikut adalah tampilan datanya:")
        st.write(df.head())

        column_name = st.text_input("Masukkan nama kolom yang ingin dianalisis:", "")
        if column_name:
            if column_name in df.columns:
                st.write(f"Melakukan analisis sentimen pada kolom: {column_name}")
                with st.spinner("Loading model dan preprocessing data..."):
                    model, vectorizer = load_model()
                    texts = preprocess_data(df, column_name)
                    df['PreprocessText'] = texts
                    features = vectorizer.transform(texts)
                
                with st.spinner("Melakukan analisis sentimen..."):
                    predictions = model.predict(features)
                    df['Sentiment'] = predictions
                st.success("Analisis sentimen selesai!")
                
                # Display results
                st.write("### Hasil Analisis Sentimen")
                st.write("**Chart Distribusi Sentimen**")
                sentiment_counts = df['Sentiment'].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Bar Chart")
                    st.bar_chart(sentiment_counts)
                with col2:
                    st.write("Pie Chart")
                    fig, ax = plt.subplots()
                    sentiment_counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                    ax.set_ylabel('')
                    st.pyplot(fig)

                st.write("**Contoh Data Hasil Analisis Sentimen**")
                st.write("5 data teratas hasil sentimen positif:")
                st.write(df[df['Sentiment'] == 'positive'].head())
                st.write("5 data teratas hasil sentimen negatif:")
                st.write(df[df['Sentiment'] == 'negative'].head())

                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download hasil analisis sentimen",
                    data=csv,
                    file_name='hasil_sentimen.csv',
                    mime='text/csv',
                )
            else:
                st.error("Nama kolom tidak ditemukan di file CSV. Mohon masukkan nama kolom yang valid.")

if __name__ == "__main__":
    main()