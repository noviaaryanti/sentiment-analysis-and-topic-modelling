import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from gensim.utils import simple_preprocess

@st.cache_resource()
def load_models():
    lda_positive = LdaModel.load('models/optimal_model_pos')
    lda_negative = LdaModel.load('models/optimal_model_neg')
    tfidf_positive = TfidfModel.load('models/tfidf_pos')
    tfidf_negative = TfidfModel.load('models/tfidf_neg')
    dictionary_positive = corpora.Dictionary.load('models/pos_dictionary')
    dictionary_negative = corpora.Dictionary.load('models/neg_dictionary')
    return lda_positive, lda_negative, tfidf_positive, tfidf_negative, dictionary_positive, dictionary_negative

# Preprocessing functions
def preprocess_data(texts):
    for text in texts:
        yield(simple_preprocess(str(text).encode('utf-8')))
    return text

# Function to get dominant topic for each document
def get_dominant_topic(ldamodel, corpus, texts):
    dominant_topics = []
    for i, row in enumerate(corpus):
        topics = ldamodel.get_document_topics(row)
        dominant_topic = max(topics, key=lambda x: x[1])
        dominant_topics.append((i, dominant_topic[0], dominant_topic[1]))

    df = pd.DataFrame(dominant_topics, columns=['Document_Index', 'Dominant_Topic', 'Probability'])
    df.set_index('Document_Index', inplace=True)
    contents = pd.Series(texts)
    df = pd.concat([df, contents], axis=1)
    df.columns = ['Dominant_Topic', 'Probability', 'Text']
    return df

def main():
    st.title("Topic Modelling")
    st.write("Di halaman ini, Anda dapat melakukan pemodelan topik pada data kendaraan listrik.")
    
    st.write("### Input Data")
    uploaded_file = st.file_uploader("Upload File CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah! Berikut adalah tampilan datanya:")
        st.write(df.head())

        column_name = st.text_input("Masukkan nama kolom yang ingin dianalisis:", "")
        sentiment_column = st.text_input("Masukkan nama kolom sentimen:", "")
        
        if column_name and sentiment_column:
            if column_name in df.columns and sentiment_column in df.columns:
                # Preprocessing text
                st.write(f"Melakukan pemodelan topik pada kolom: {column_name}")
                with st.spinner("Loading model dan preprocessing data..."):
                    pos_data = df[df[sentiment_column] == "positive"][column_name]
                    pos_data.reset_index(drop=True, inplace=True)
                    neg_data = df[df[sentiment_column] == "negative"][column_name]
                    neg_data.reset_index(drop=True, inplace=True)
                    pos_preprocess = preprocess_data(pos_data)
                    neg_preprocess = preprocess_data(neg_data)
                    # Load Model
                    lda_positive, lda_negative, tfidf_positive, tfidf_negative, dictionary_positive, dictionary_negative = load_models()
                
                with st.spinner("Pemodelan topik sedang diproses..."):
                    pos_bow = [dictionary_positive.doc2bow(doc) for doc in pos_preprocess]
                    neg_bow = [dictionary_negative.doc2bow(doc) for doc in neg_preprocess]
                    positive_corpus = tfidf_positive[pos_bow]
                    negative_corpus = tfidf_negative[neg_bow]
                    positive_dominant_topics = get_dominant_topic(lda_positive, positive_corpus, pos_data)
                    negative_dominant_topics = get_dominant_topic(lda_negative, negative_corpus, neg_data)
                st.success("Pemodelan topik selesai!")
                
                st.write("### Hasil Pemodelan Topik")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Pemodelan Topik Sentimen Positif")
                    
                    st.write("**Kata kunci untuk tiap topik Sentimen Positif:**")
                    for idx, topic in enumerate(lda_positive.print_topics()):
                        st.write(f"Topik {idx}: {topic}")
                    
                    st.write("**Topik dominan untuk Sentimen Positif:**")
                    st.write(positive_dominant_topics.head())
                    
                    st.write("**Distribusi Topik Dominan untuk Sentimen Positif:**")
                    fig, ax = plt.subplots()
                    sns.countplot(x='Dominant_Topic', data=positive_dominant_topics, ax=ax)
                    st.pyplot(fig)
                    
                    csv1 = positive_dominant_topics.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Hasil Topik Positif",
                        data=csv1,
                        file_name='hasil_topik_positif.csv',
                        mime='text/csv',
                    )
                with col2:
                    st.write("#### Pemodelan Topik Sentimen Negatif")
                    
                    st.write("**Kata kunci untuk tiap topik Sentimen Negatif:**")
                    for idx, topic in enumerate(lda_negative.print_topics()):
                        st.write(f"Topik {idx}: {topic}")
                
                    st.write("**Topik dominan untuk Sentimen Negatif:**")
                    st.write(negative_dominant_topics.head())
                
                    st.write("**Distribusi Topik Dominan untuk Sentimen Negatif:**")
                    fig, ax = plt.subplots()
                    sns.countplot(x='Dominant_Topic', data=negative_dominant_topics, ax=ax)
                    st.pyplot(fig)
                
                    csv2 = negative_dominant_topics.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Hasil Topik Negatif",
                        data=csv2,
                        file_name='hasil_topik_negatif.csv',
                        mime='text/csv',
                    )
            else:
                st.error("Nama kolom teks atau sentimen tidak ditemukan dalam data yang diunggah.")
        else:
            st.info("Masukkan nama kolom teks dan sentimen untuk melanjutkan.")
    else:
        st.info("Silakan unggah file CSV untuk memulai.")
                
if __name__ == "__main__":
    main()