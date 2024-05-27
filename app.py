import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title = "Aplikasi Analisis Sentimen dan Pemodelan Topik",
    page_icon = "📊"
)

# Horizontal menu
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Sentiment Analysis", "Topic Modelling"],  # required
    icons=["house", "emoji-neutral", "book"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
)

if selected == "Home":
    st.title("Aplikasi Analisis Sentimen dan Pemodelan Topik Kendaraan Listrik")
    st.write("Aplikasi web ini memungkinkan Anda untuk melakukan analisis sentimen dan pemodelan topik yang difokuskan pada data kendaraan listrik.")
    
    st.write("### Menu Aplikasi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("""**Home**
               \nBerisi penjelasan tentang aplikasi""")
    with col2:
        st.write("""**Sentimen Analysis**
               \nMemungkinkan Anda melakukan klasfikasi sentimen berdasarkan berdasarkan kumpulan data yang diinputkan pada sistem. Data yang diinputkan hanya dapat dalam bentuk format csv.  Kemudian sistem akan memberikan keluaran atau output berupa dokumen yang telah diklasifikasi ke dalam sentimen positif dan negatif.""")
    with col3:
        st.write("""**Topic Modelling**
               \nMemungkinkan Anda untuk melakukan pemodelan topik berdasarkan kumpulan data yang diinputkan pada sistem. Data yang diinputkan hanya dapat dalam bentuk format csv. yang berisi kolom teks serta kolom sentimen (yang mengelompokkan teks ke dalam sentimen positif dan negatif). Kemudian sistem akan memberikan keluaran atau output berupa topik – topik yang banyak muncul pada data.""")

elif selected == "Sentiment Analysis":
    from pages import Sentiment_Analysis
    Sentiment_Analysis.main()

elif selected == "Topic Modelling":
    from pages import Topic_Modelling
    Topic_Modelling.main()