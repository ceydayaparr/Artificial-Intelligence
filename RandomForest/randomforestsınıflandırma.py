import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import altair as alt

# Veriyi yükle
df = pd.read_csv(r'C:\Users\yavuz\PycharmProjects\AIM_P2\WineQT.csv')

# Kalite eşiğini belirle ve quality_label sütununu oluştur
quality_threshold = 6
df['quality_label'] = ['kaliteli' if quality >= quality_threshold else 'kalitesiz' for quality in df['quality']]

# 'Id' sütununu düşür
df.drop(['Id'], axis=1, inplace=True)

# Özellik matrisini (X) ve hedef değişkeni (y) oluştur
X = df.drop('quality_label', axis=1)
y = df['quality_label']

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest sınıflandırıcısı modelini oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test kümesi üzerinde tahminler yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu hesapla ve yazdır
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Sınıflandırma raporunu oluştur ve yazdır
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Karışıklık matrisini hesapla ve ısı haritası olarak görselleştir
cm = confusion_matrix(y_test, y_pred)

# Karışıklık Matrisi Heatmap
cm_df = pd.DataFrame(cm, index=['Kalitesiz (Gerçek)', 'Kaliteli (Gerçek)'], columns=['Kalitesiz (Tahmin)', 'Kaliteli (Tahmin)'])
heatmap_chart = alt.Chart(cm_df.reset_index().melt(id_vars='index')).mark_rect().encode(
    x=alt.X('variable:O', axis=alt.Axis(title='Tahmin')),
    y=alt.Y('index:O', axis=alt.Axis(title='Gerçek')),
    color=alt.Color('value:Q', scale=alt.Scale(scheme='blues')),
    tooltip=['value:Q']
).properties(
    title='Karışıklık Matrisi (Random Forest)'
).interactive()

# Heatmap üzerindeki sayıları ekle
text = heatmap_chart.mark_text(baseline='middle').encode(
    text='value:Q',
    color=alt.condition(alt.datum.value > 100, alt.value('white'), alt.value('black'))
)
chart = heatmap_chart + text
chart.save('confusion_matrix_heatmap_random_forest.json')

# Kaliteli ve kalitesiz şarapların sayısını ve yüzdesini hesapla ve yazdır
quality_counts = df['quality_label'].value_counts()
quality_percentage = (quality_counts['kaliteli'] / quality_counts.sum()) * 100
print("\nQuality Metrics:")
print(quality_counts.rename_axis('Quality').rename('Total Wines').to_markdown(numalign="left", stralign="left"))
print(f'Percentage of High Quality Wines: {quality_percentage:.2f}%')

# Feature İmportances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

# Feature İmportances Bar Chart
chart = alt.Chart(feature_importances).mark_bar().encode(
    x=alt.X('Importance:Q', axis=alt.Axis(title='Importance')),
    y=alt.Y('Feature:N', sort='-x'),
    tooltip=['Feature', 'Importance']
).properties(title='Feature Importances (Random Forest)').interactive()

chart.save('feature_importances_bar_chart_random_forest.json')
