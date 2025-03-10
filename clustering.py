import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Kmeans kümeleme

# Veri kümesini yükle
file_path = 'WineQT.csv'
data = pd.read_csv(file_path)

# Id sütununu düşür (modelleme için gereksiz)
data = data.drop(columns=['Id'])

# Özellikleri ölçeklendirme (Standardizasyon)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# K-means kümeleme
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Küme etiketlerini veri kümesine ekle
data['Cluster'] = kmeans.labels_

# Küme merkezlerini görselleştirme
centers = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers, columns=data.columns[:-1])
center_df['Cluster'] = range(3)

# Sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(data['alcohol'], data['quality'], c=data['Cluster'], cmap='viridis')
plt.scatter(center_df['alcohol'], center_df['quality'], c='red', marker='X', s=200)
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Şarap Verileri Kümeleme Analizi')
plt.show()

