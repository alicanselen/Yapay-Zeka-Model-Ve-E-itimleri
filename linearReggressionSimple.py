import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Rastgele bir veri seti oluşturma
np.random.seed(0)  # Rastgele sayı üreticisi için sabit bir başlangıç değeri belirleyin
X = 2 * np.random.rand(100, 1)  # 100 adet rastgele X değeri oluşturun (0-2 arası)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + rastgele gürültü ekleyin

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini oluşturma
model = LinearRegression()
model.fit(X_train, y_train)  # Modeli eğitim verisiyle eğitin

# Modeli test verisi ile kullanarak tahmin yapma
y_pred = model.predict(X_test)  # Test verisi üzerinde tahminler yapın

# Modelin performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)  # Ortalama Kare Hatasını hesaplayın
r2 = r2_score(y_test, y_pred)  # R^2 skorunu hesaplayın

# Performans metriklerini ekrana yazdırma
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Sonuçları görselleştirme
plt.scatter(X_test, y_test, color='blue', label='Gerçek Değerler')  # Test verisinin gerçek değerlerini çizdirin
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Tahminler')  # Modelin tahminlerini çizdirin
plt.xlabel('Bağımsız Değişken (X)')  # X eksenini etiketleyin
plt.ylabel('Bağımlı Değişken (y)')  # Y eksenini etiketleyin
plt.legend()  # Grafik için bir açıklama ekleyin
plt.show()  # Grafiği gösterin
