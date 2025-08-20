import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
import google_play_scraper
from google_play_scraper import app, Sort, reviews
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download stopwords jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.strip()
    text = re.sub(r"http\S+", "[URL]", text)        # ganti url
    text = re.sub(r"@\w+", "[USER]", text)          # ganti mention
    text = re.sub(r"\d+", "[NUM]", text)            # ganti angka
    return text

# Load model dan tokenizer
@st.cache_resource
def load_model():
    model_path = "indobert-sentiment"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Fungsi prediksi sentimen
def predict_sentiment(text, model, tokenizer):
    # Bersihkan teks terlebih dahulu
    cleaned_text = clean_text(text)
    
    # Tokenisasi dan prediksi
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return ["negatif", "netral", "positif"][prediction]

# Fungsi untuk memproses batch teks
def process_batch(texts, model, tokenizer):
    results = []
    for text in texts:
        if text.strip():  # Skip empty lines
            sentiment = predict_sentiment(text, model, tokenizer)
            results.append({"text": text, "sentiment": sentiment})
    return results

# Fungsi untuk scraping ulasan dari Play Store
def scrape_playstore(app_id, count=100, selected_sort=Sort.NEWEST):

    try:
        # Gunakan reviews() bukan reviews_all() untuk lebih cepat
        result, _ = google_play_scraper.reviews(
            app_id,
            lang='id',  # Bahasa Indonesia
            country='id',
            sort=selected_sort,
            count=count
        )
        reviews = [review['content'] for review in result if review['content']]
        return reviews[:100]  # Batasi maksimal 100 ulasan
    except Exception as e:
        st.error(f"Error saat scraping: {e}")
        return []

# Fungsi untuk membuat wordcloud
def create_wordcloud(texts, sentiment=None):
    if not texts:
        return None
    
    # Gabungkan semua teks
    if sentiment:
        # Filter teks berdasarkan sentimen jika ada
        combined_text = " ".join([item["text"] for item in texts if item["sentiment"] == sentiment])
    else:
        # Gunakan semua teks jika tidak ada filter sentimen
        combined_text = " ".join([item["text"] for item in texts])
    
    if not combined_text.strip():
        return None
    
    # Stopwords bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))
    
    # Tambahkan stopwords kustom
    custom_stopwords = {"yang", "dengan", "dan", "di", "ke", "ini", "itu", "dari", "untuk", "pada", "adalah", "dalam", "tidak", "ada"}
    stop_words.update(custom_stopwords)
    
    # Buat wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        colormap='viridis',
        collocations=False
    ).generate(combined_text)
    
    return wordcloud

# Fungsi untuk menganalisis permasalahan dalam teks
def analyze_issues(texts):
    if not texts:
        return {}
    
    # Kata kunci untuk setiap kategori permasalahan
    issue_keywords = {
        'login': ['login', 'masuk', 'akun', 'daftar', 'registrasi', 'password', 'kata sandi', 'lupa', 'verifikasi', 'otp'],
        'bug': ['bug', 'error', 'crash', 'hang', 'macet', 'berhenti', 'tidak berfungsi', 'tidak bisa', 'gagal', 'bermasalah', 'force close'],
        'tampilan': ['tampilan', 'interface', 'desain', 'layout', 'tema', 'warna', 'font', 'ukuran', 'tata letak', 'tombol', 'ikon','bingung']

    }
    
    # Hitung kemunculan kata kunci untuk setiap kategori
    issue_counts = {category: 0 for category in issue_keywords}
    issue_texts = {category: [] for category in issue_keywords}
    issue_items = {category: [] for category in issue_keywords}  # Menyimpan item lengkap untuk analisis sentimen
    
    for item in texts:
        text = item["text"].lower()
        for category, keywords in issue_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    issue_counts[category] += 1
                    issue_texts[category].append(item["text"])
                    issue_items[category].append(item)  # Simpan item lengkap termasuk sentimen
                    break  # Hanya hitung sekali per kategori per teks
    
    # Hitung distribusi sentimen untuk setiap kategori
    issue_sentiments = {}
    for category, items in issue_items.items():
        if items:
            sentiments = [item["sentiment"] for item in items]
            issue_sentiments[category] = {
                "negatif": sentiments.count("negatif"),
                "netral": sentiments.count("netral"),
                "positif": sentiments.count("positif")
            }
        else:
            issue_sentiments[category] = {"negatif": 0, "netral": 0, "positif": 0}
    
    return {'counts': issue_counts, 'texts': issue_texts, 'items': issue_items, 'sentiments': issue_sentiments}

# Fungsi untuk visualisasi hasil
def visualize_results(results):
    if not results:
        return
    
    # Hitung jumlah sentimen
    sentiments = [r["sentiment"] for r in results]
    sentiment_counts = {"negatif": sentiments.count("negatif"), 
                       "netral": sentiments.count("netral"), 
                       "positif": sentiments.count("positif")}
    
    # Buat visualisasi - diagram batang sentimen
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Warna sesuai permintaan: merah untuk negatif, abu-abu untuk netral, hijau untuk positif
    colors = ['#FF0000', '#808080', '#00FF00']
    
    # Bar chart
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors)
    ax.set_title('Distribusi Sentimen')
    ax.set_ylabel('Jumlah')
    ax.set_xlabel('Kategori Sentimen')
    
    # Tambahkan nilai di atas bar
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        ax.text(i, count + 0.5, str(count), ha='center')
    
    st.pyplot(fig)
    
    # Analisis permasalahan
    issues = analyze_issues(results)
    if issues and any(issues['counts'].values()):
        st.subheader("Analisis Permasalahan")
        
        # Visualisasi jumlah permasalahan
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(issues['counts'].keys())
        counts = list(issues['counts'].values())
        
        # Warna untuk kategori permasalahan
        issue_colors = ['#FF9999', '#99CCFF', '#FFCC99']
        
        # Bar chart permasalahan
        bars = ax.bar(categories, counts, color=issue_colors)
        ax.set_title('Distribusi Permasalahan')
        ax.set_ylabel('Jumlah Ulasan')
        ax.set_xlabel('Kategori Permasalahan')
        
        # Tambahkan nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Tampilkan detail permasalahan
        for category, texts in issues['texts'].items():
            if texts:
                # Hitung distribusi sentimen untuk kategori ini
                category_sentiments = issues['sentiments'][category]
                
                # Gunakan expander untuk menampilkan/menyembunyikan ulasan
                with st.expander(f"Detail Permasalahan {category.capitalize()} ({len(texts)} ulasan)"):
                    # Tampilkan distribusi sentimen untuk kategori ini
                    st.subheader(f"Distribusi Sentimen untuk {category.capitalize()}")
                    
                    # Buat visualisasi - diagram batang sentimen untuk kategori ini
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Warna sesuai permintaan: merah untuk negatif, abu-abu untuk netral, hijau untuk positif
                    colors = ['#FF0000', '#808080', '#00FF00']
                    
                    # Bar chart
                    sentiment_labels = ["negatif", "netral", "positif"]
                    sentiment_values = [category_sentiments[label] for label in sentiment_labels]
                    
                    bars = ax.bar(sentiment_labels, sentiment_values, color=colors)
                    ax.set_title(f'Distribusi Sentimen - {category.capitalize()}')
                    ax.set_ylabel('Jumlah')
                    ax.set_xlabel('Kategori Sentimen')
                    
                    # Tambahkan nilai di atas bar
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Tampilkan semua ulasan dengan label sentimen
                    st.subheader("Daftar Ulasan dengan Label Sentimen:")
                    
                    # Buat tabel untuk menampilkan ulasan dan label sentimen
                    issue_data = []
                    for item in issues['items'][category]:
                        issue_data.append({
                            "Komentar": item["text"],
                            "Sentimen": item["sentiment"]
                        })
                    
                    if issue_data:
                        issue_df = pd.DataFrame(issue_data)
                        st.dataframe(issue_df, use_container_width=True)
                    else:
                        st.info("Tidak ada ulasan untuk kategori ini.")
        
        # Tampilkan semua ulasan langsung
        st.subheader("Daftar Ulasan:")
        for i, text in enumerate(texts, 1):
            st.write(f"{i}. {text}")
    
    # Wordcloud untuk setiap kategori permasalahan
    st.subheader("WordCloud Permasalahan")
    cols = st.columns(len(issues['texts']))
    
    for i, (category, texts) in enumerate(issues['texts'].items()):
        if texts:
            with cols[i]:
                st.write(f"**{category.capitalize()}**")
                # Buat wordcloud khusus untuk kategori permasalahan
                issue_text = " ".join(texts)
                if issue_text.strip():
                    # Stopwords bahasa Indonesia
                    stop_words = set(stopwords.words('indonesian'))
                    custom_stopwords = {"yang", "dengan", "dan", "di", "ke", "ini", "itu", "dari", "untuk", "pada", "adalah", "dalam", "tidak", "ada"}
                    stop_words.update(custom_stopwords)
                    
                    wordcloud = WordCloud(
                        width=400, 
                        height=300, 
                        background_color='white',
                        stopwords=stop_words,
                        max_words=50,
                        colormap='Blues' if category == 'login' else 'Reds' if category == 'bug' else 'Greens',
                        collocations=False
                    ).generate(issue_text)
                    
                    plt.figure(figsize=(5, 4))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
    
    # Tambahkan wordcloud untuk semua teks
    st.subheader("WordCloud - Semua Teks")
    wordcloud = create_wordcloud(results)
    if wordcloud:
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.info("Tidak cukup data untuk membuat wordcloud")
    
    # Tambahkan wordcloud untuk setiap sentimen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("WordCloud - Negatif")
        neg_wordcloud = create_wordcloud(results, "negatif")
        if neg_wordcloud:
            plt.figure(figsize=(8, 4))
            plt.imshow(neg_wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Tidak ada teks negatif")
    
    with col2:
        st.subheader("WordCloud - Netral")
        neu_wordcloud = create_wordcloud(results, "netral")
        if neu_wordcloud:
            plt.figure(figsize=(8, 4))
            plt.imshow(neu_wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Tidak ada teks netral")
    
    with col3:
        st.subheader("WordCloud - Positif")
        pos_wordcloud = create_wordcloud(results, "positif")
        if pos_wordcloud:
            plt.figure(figsize=(8, 4))
            plt.imshow(pos_wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Tidak ada teks positif")

# Main app
def main():
    st.title("Analisis Sentimen Bahasa Indonesia")
    st.write("Aplikasi ini menggunakan model IndoBERT untuk menganalisis sentimen teks dalam Bahasa Indonesia.")
    
    # Load model
    model, tokenizer = load_model()
    
    # Buat tabs
    tab1, tab2, tab3 = st.tabs(["Input Manual", "Input File", "Scraping Play Store"])
    
    # Tab 1: Input Manual
    with tab1:
        st.header("Input Komentar Manual")
        st.write("Masukkan komentar (maksimal 100 baris)")
        
        text_input = st.text_area("Komentar", height=300)
        
        if st.button("Analisis Sentimen", key="analyze_manual"):
            if text_input:
                # Split input by lines and limit to 100
                lines = text_input.strip().split('\n')[:100]
                
                with st.spinner("Menganalisis sentimen..."):
                    results = process_batch(lines, model, tokenizer)
                    
                    # Tampilkan hasil
                    st.subheader("Hasil Analisis")
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df)
                    
                    # Visualisasi
                    visualize_results(results)
            else:
                st.warning("Silakan masukkan teks untuk dianalisis.")
    
    # Tab 2: Input File
    with tab2:
        st.header("Input File CSV")
        st.write("Upload file CSV dengan kolom teks (maksimal 100 baris)")
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        text_column = st.text_input("Nama kolom teks", "text")
        
        if uploaded_file and st.button("Analisis Sentimen", key="analyze_file"):
            try:
                df = pd.read_csv(uploaded_file)
                
                if text_column not in df.columns:
                    st.error(f"Kolom '{text_column}' tidak ditemukan dalam file CSV. Kolom yang tersedia: {', '.join(df.columns)}")
                else:
                    # Limit to 100 rows
                    texts = df[text_column].astype(str).tolist()[:100]
                    
                    with st.spinner("Menganalisis sentimen..."):
                        results = process_batch(texts, model, tokenizer)
                        
                        # Tampilkan hasil
                        st.subheader("Hasil Analisis")
                        result_df = pd.DataFrame(results)
                        st.dataframe(result_df)
                        
                        # Visualisasi
                        visualize_results(results)
            except Exception as e:
                st.error(f"Error saat memproses file: {e}")
    
    # Tab 3: Scraping Play Store
    with tab3:
        st.header("Scraping Ulasan Play Store")
        st.write("Masukkan ID aplikasi Google Play Store untuk mengambil ulasan (maksimal 100 ulasan)")
        
        app_id = st.text_input("ID Aplikasi Play Store", "com.gojek.app")
        count = st.slider("Jumlah ulasan", min_value=1, max_value=100, value=50)
        sort_option = st.selectbox(
            "Urutkan berdasarkan",
            options=["Terbaru", "Relevan"],
            index=0
        )
        
        # Mapping pilihan ke enum Sort
        sort_mapping = {
            "Terbaru": Sort.NEWEST,
            "Relevan": Sort.MOST_RELEVANT
        }
        selected_sort = sort_mapping[sort_option]
        
        if st.button("Ambil Ulasan dan Analisis", key="scrape_analyze"):
            if app_id:
                with st.spinner("Mengambil ulasan dari Play Store..."):
                    reviews = scrape_playstore(app_id, count, selected_sort)

                    
                    if reviews:
                        st.success(f"Berhasil mengambil {len(reviews)} ulasan")
                        
                        # Tampilkan beberapa ulasan sebagai contoh
                        st.subheader("Contoh Ulasan")
                        for i, review in enumerate(reviews[:5]):
                            st.text(f"{i+1}. {review[:100]}..." if len(review) > 100 else f"{i+1}. {review}")
                        
                        # Analisis sentimen
                        with st.spinner("Menganalisis sentimen ulasan..."):
                            results = process_batch(reviews, model, tokenizer)
                            
                            # Tampilkan hasil
                            st.subheader("Hasil Analisis")
                            result_df = pd.DataFrame(results)
                            st.dataframe(result_df)
                            
                            # Visualisasi
                            visualize_results(results)
                    else:
                        st.warning("Tidak ada ulasan yang berhasil diambil. Periksa ID aplikasi dan coba lagi.")
            else:
                st.warning("Silakan masukkan ID aplikasi Play Store.")

if __name__ == "__main__":
    main()