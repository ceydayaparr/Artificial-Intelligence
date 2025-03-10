import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import altair as alt
import numpy as np

# Tüm satır ve sütunları göstermek için pandas ayarları
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# CSV dosyasını DataFrame olarak oku
df = pd.read_csv(r'C:\Users\yavuz\PycharmProjects\AIM_P2\WineQT.csv')

# İlk 5 satırı göster
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Sütun adlarını ve veri türlerini yazdır
print(df.info())

# Eşik 6'ya göre `quality_label` sütunu oluştur
df['quality_label'] = df['quality'].apply(lambda x: 'kaliteli' if x >= 6 else 'kalitesiz')

# `quality` ve `Id` sütunlarını düşür
df.drop(['quality', 'Id'], axis=1, inplace=True)

# Özellik matrisi (X) ve hedef vektörü (y) oluştur
X = df.drop('quality_label', axis=1)
y = df['quality_label']

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test kümesi üzerinde tahmin yap
y_pred = model.predict(X_test)

# Karışıklık matrisini hesapla
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Karışıklık matrisi için DataFrame oluştur
cm_df = pd.DataFrame(cm, index=['Actual Kalitesiz', 'Actual Kaliteli'], columns=['Predicted Kalitesiz', 'Predicted Kaliteli'])

# Isı haritası oluştur
heatmap = alt.Chart(cm_df.stack().reset_index().rename(columns={'level_0': 'Actual', 'level_1': 'Predicted', 0: 'Count'})).mark_rect().encode(
    x=alt.X('Predicted:O', axis=alt.Axis(title='Predicted')),
    y=alt.Y('Actual:O', axis=alt.Axis(title='Actual')),
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
    tooltip=['Actual', 'Predicted', 'Count']
).properties(
    title='Confusion Matrix for Random Forest Model'
).interactive()

# Isı haritasına metin etiketleri ekle
text = heatmap.mark_text(baseline='middle').encode(
    text='Count:Q',
    color=alt.condition(alt.datum.Count > np.mean(cm), alt.value('white'), alt.value('black'))
)

# Isı haritası ve metin katmanlarını birleştir
chart = heatmap + text

# Grafiği kaydet
chart.save('confusion_matrix_heatmap_random_forest.json')
