import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veriyi yükle
df = pd.read_csv(r'C:\Users\yavuz\PycharmProjects\AIM_P2\WineQT.csv')

# Kalite eşiğini belirle ve quality_label sütununu oluştur
quality_threshold = 6
df['quality_label'] = ['kaliteli' if quality >= quality_threshold else 'kalitesiz' for quality in df['quality']]

# Özellik matrisini (X) ve hedef değişkeni (y) oluştur
X = df[['pH', 'alcohol', 'sulphates']]
y = df['quality_label']

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Modelin doğruluğunu hesapla ve yazdır
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Kullanıcının şarap bilgileri (örnek değerler)
user_wine = pd.DataFrame([[3, 10.5, 0.6]], columns=['pH', 'alcohol', 'sulphates'])

# Tahmin yap
prediction = model.predict(user_wine)

# Tahmin edilen kaliteyi ekrana yazdır
print("Tahmin edilen şarap kalitesi:", prediction[0])
