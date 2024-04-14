import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
import matplotlib.pyplot as plt
#author: github.com/ceyhunemre0

metinler = [
    "Bu bir örnek metindir. Türkçe metin doğrulama için kullanılabilir.",
    "Bu metin doğrulama örneği için bir diğer örnek metindir.",
    "Metinler arasındaki benzerliği ölçmek için Türkçe metinler gereklidir.",
    "Metin doğrulama algoritmalarını test etmek için birkaç farklı örnek metin bulunmalıdır.",
    "Doğal Dil İşleme projelerinde kullanılmak üzere çeşitli Türkçe metinler toplanmalıdır."
]

metinler = [filter(metin) for metin in metinler]
metinler = [remove_stopwords(metin) for metin in metinler]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(metinler)


# Benzerlik matrisini hesapla ve görselleştir
similarity_matrix = cosine_similarity(X, X)
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Benzerlik')
plt.title('Benzerlik Matrisi')
plt.xlabel('Örnek Numarası')
plt.ylabel('Örnek Numarası')
plt.xticks(ticks=np.arange(5))
plt.yticks(ticks=np.arange(5))
plt.grid(False)

print(similarity_matrix) # Benzerlik matrisini ekrana yazdır
plt.show() # Benzerlik matrisini görselleştirilmiş şekilde ekrana çizdir



# filtreleme ve stop words kaldırma

stop_words = set(stopwords.words('turkish'))
def filter(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def remove_stopwords(text):
    words = word_tokenize(text, "turkish")
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)