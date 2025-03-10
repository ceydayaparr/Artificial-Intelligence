import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# CSV dosyasını oku
veri = pd.read_csv('WineQT.csv')

# İlk 5 satırı göster
print("Veri setinden ilk 5 satır:")
print(veri.head().to_markdown(index=False, numalign="left", stralign="left"))

# Sütun adlarını ve veri tiplerini yazdır
print("\nSütun Adları ve Veri Tipleri:")
print(veri.info())

# 'Id' sütunu hariç korelasyon matrisini hesapla
korelasyon_matrisi = veri.drop(columns=['Id']).corr().round(3)

print("\nKorelasyon Matrisi:")
print(korelasyon_matrisi.to_markdown(numalign="left", stralign="left"))

# 'quality' ile diğer değişkenlerin korelasyonlarını (mutlak değer) bul ve sırala
kalite_korelasyonlari = korelasyon_matrisi['quality'].drop('quality').abs().sort_values(ascending=False)
print("\nKalite ile Diğer Değişkenlerin Korelasyonları (Mutlak Değer):")
print(kalite_korelasyonlari.to_markdown(numalign="left", stralign="left"))

# Load the dataset
wine_data = veri

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = wine_data.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title of the heatmap
plt.title('Heatmap of Wine Quality Dataset Correlations')
plt.show()
