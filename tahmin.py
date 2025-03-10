import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import altair as alt

# Veriyi yükle
df = pd.read_csv('WineQT.csv')

# Kalite eşiğini belirle ve quality_label sütununu oluştur
quality_threshold = 6
df['quality_label'] = ['kaliteli' if quality >= quality_threshold else 'kalitesiz' for quality in df['quality']]

# Özellik matrisini (X) ve hedef değişkeni (y) oluştur
X = df[['pH', 'alcohol', 'sulphates']]
y = df['quality']  # Hedef değişken olarak quality kullanılacak

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR modelini oluştur ve eğit
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Modelin performansını değerlendir (MSE ve MAE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Kullanıcının şarap bilgileri (örnek değerler)
user_wine = [[3.5, 10.5, 0.6]]

# DataFrame'e dönüştürme
user_wine_df = pd.DataFrame(user_wine, columns=['pH', 'alcohol', 'sulphates'])

# Tahmin yapma
prediction = model.predict(user_wine_df)

# Tahmin edilen kaliteyi ekrana yazdır
print("Tahmin edilen şarap kalitesi (puan):", prediction[0])

# Grafik Çizimi

# pH Değeri Dağılımı - Kaliteye Göre
chart1 = alt.Chart(df).mark_boxplot().encode(
    x='quality_label:N',
    y='pH:Q',
    color='quality_label:N'
).properties(
    title='pH Değeri Dağılımı - Kaliteye Göre'
).interactive()

chart1.save('ph_dagilimi_kaliteye_gore_boxplot.json')

# Sülfat Oranı Dağılımı - Kaliteye Göre
chart2 = alt.Chart(df).mark_boxplot().encode(
    x='quality_label:N',
    y='sulphates:Q',
    color='quality_label:N'
).properties(
    title='Sülfat Oranı Dağılımı - Kaliteye Göre'
).interactive()

chart2.save('sulfat_orani_dagilimi_kaliteye_gore_boxplot.json')

# Alkol Oranı Dağılımı - Kaliteye Göre
chart3 = alt.Chart(df).mark_boxplot().encode(
    x='quality_label:N',
    y='alcohol:Q',
    color='quality_label:N'
).properties(
    title='Alkol Oranı Dağılımı - Kaliteye Göre'
).interactive()

chart3.save('alkol_orani_dagilimi_kaliteye_gore_boxplot.json')

# pH, Sülfat ve Alkol Oranları Arasındaki İlişki
chart4 = alt.Chart(df).mark_circle().encode(
    x='pH:Q',
    y='sulphates:Q',
    size='alcohol:Q',
    color='quality_label:N',
    tooltip=['pH:Q', 'sulphates:Q', 'alcohol:Q', 'quality_label:N']
).properties(
    title='pH, Sülfat ve Alkol Oranları Arasındaki İlişki'
).interactive()

chart4.save('ph_sulfat_alkol_iliskisi_scatter_plot.json')
