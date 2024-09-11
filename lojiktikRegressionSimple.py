import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.corpus import stopwords
import re

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dataset.csv')

dataset = dataset.drop(columns='B')

dataset.drop_duplicates(subset="Body" , keep= False , inplace= True)

def optimizasyon(dataset):
    dataset = dataset.dropna() #bos veri iceren verileri siler

    stop_words = set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)

    for ind in dataset.index:
        body = dataset['Body'][ind]
        body = body.lower()
        body = re.sub(r'http\S+', '', body)#url kaldır
        body = re.sub('\[[^]]*\]', '', body)  # Köşeli parantez içeriğini kaldır
        body = (" ").join([word for word in body.split() if not word in stop_words]) # Stopwords'leri kaldır
        body = "".join([char for char in body if not char in noktalamaIsaretleri])  # Noktalama işaretlerini kaldır
        dataset['Body'][ind] = body # Temizlenmiş metni geri yaz
    return dataset

dataset = optimizasyon(dataset)

# Label değerine göre veri setini ayır

yorumlar_makina = dataset[dataset['Label']==0]
yorumlar_insan = dataset[dataset['Label']==1]

yorumlar_insan

tfIdf = TfidfVectorizer(binary=False , ngram_range=(1,3))

makina_vec = tfIdf.fit_transform(yorumlar_makina['Body'].tolist())
insan_vec = tfIdf.fit_transform(yorumlar_insan['Body'].tolist())

x = dataset['Body']
y = dataset['Label']

x_vec = tfIdf.fit_transform(x)

# Eğitim ve test veri setlerine ayır
x_egitim_vec, x_test_vec, y_egitim, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=0)

# Lojistik regresyon modeli oluştur ve eğit
lojistikRegresyon = LogisticRegression()
lojistikRegresyon.fit(x_egitim_vec,y_egitim)

# Test veri seti üzerinde tahmin yap
y_tahmin = lojistikRegresyon.predict(x_test_vec)

# Eğitilmiş modeli dosyaya kaydet
pickle.dump(lojistikRegresyon, open("egitilmis_model", 'wb'))
print("Lojistik Regresyon modeli eğitildi ve kayıt edildi !")

# TF-IDF vektörleştirici modelini dosyaya kaydet
pickle.dump(tfIdf, open("vektorlestirici", 'wb'))
print("Tf-Idf vektörleştirici modeli kayıt edildi !")
# Sonuçları yazdır
print(confusion_matrix(y_test,y_tahmin))
print(classification_report(y_test,y_tahmin))
exit()
