# --- Bagian install (jika diperlukan) ---
import subprocess
import sys

def install_if_needed(pkg_name, import_name=None):
    try:
        if import_name:
            __import__(import_name)
        else:
            __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

install_if_needed("nltk")
install_if_needed("Sastrawi")

# --- Import normal setelahnya ---
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pickle
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Inisialisasi Stemmer dan stopword
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_ind = set(nltk.corpus.stopwords.words('indonesian'))

# Contoh kamus normalisasi
normalization_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "tdk": "tidak",
    "dr": "dari",
    "dgn": "dengan",
    "aja": "saja",
    "jg": "juga",
    # Tambahin sesuai kebutuhan
}

def normalize_text(text):
    tokens = text.split()
    return ' '.join([normalization_dict.get(token, token) for token in tokens])

# --- Load model dan vectorizer ---
try:
    with open('model/model_textblob.sav', 'rb') as f:
        model = pickle.load(f)
    with open('model/tfidf_textblob.sav', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("Model dan TF-IDF Vectorizer berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file model dan vectorizer ada.")
    exit()

# --- Praproses lengkap ---
def preprocess_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r"http\S+|www.\S+", "", text)  # Hapus URL
    text = re.sub(r"\d+", "", text)  # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    text = normalize_text(text)  # Normalisasi kata
    tokens = word_tokenize(text)  # Tokenisasi
    tokens = [word for word in tokens if word not in stopwords_ind]  # Stopword removal
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)

# --- Prediksi Sentimen ---
def predict_sentiment(tweet_text):
    processed_tweet = preprocess_text(tweet_text)
    tweet_vector = tfidf_vectorizer.transform([processed_tweet])
    prediction = model.predict(tweet_vector)

    if prediction[0] == 0:
        return "Negatif"
    elif prediction[0] == 1:
        return "Positif"
    else:
        return f"Label tidak dikenal: {prediction[0]}"

# --- CLI Manual ---
if __name__ == "__main__":
    print("\n--- Prediksi Sentimen Cuitan Baru ---")
    while True:
        new_tweet = input("Masukkan cuitan baru (ketik 'exit' untuk keluar): \n")
        if new_tweet.lower() == 'exit':
            break

        if new_tweet.strip() == "":
            print("Cuitan tidak boleh kosong. Silakan coba lagi.")
            continue

        sentiment = predict_sentiment(new_tweet)
        print(f"Cuitan: '{new_tweet}'")
        print(f"Sentimen Prediksi: {sentiment}\n")
