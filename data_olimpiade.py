import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('olympians.csv')
print(data)

#diagram bar masing masing bidang olahraga
data['sport'].value_counts().plot.bar()
plt.title("Distribusi Olahraga Olimpiade")
plt.xlabel("sport")
plt.ylabel("Jumlah Atlet")
plt.show()

#10 negara dengan atlet terbanyak
top_countries = data['nationality'].value_counts().head(10)
top_countries.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("10 Negara dengan Atlet Terbanyak")
plt.ylabel("")
plt.show()

#spider chart jumlah medali USA
data_grouped = data.groupby('nationality').agg({
    'gold': 'sum', 'silver': 'sum', 'bronze': 'sum'}).reset_index()
data_grouped = data_grouped[data_grouped['nationality'] == 'USA'].iloc[0]

labels = ['gold', 'silver', 'bronze']
values = data_grouped[labels].values.tolist()

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
values += values[:1]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.2)

ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
ax.set_title("Spider Chart Medali USA")
plt.show()

#5 atlet dan asal negaranya dengan emas terbanyak
tabel_medali = data[["name", "nationality", "gold"]].sort_values(by="gold", ascending=False).head(5)
print("Atlet dan Asal Negaranya dengan Emas Terbanyak:")
print(tabel_medali.to_string(index=False))

#scatter plot fisik atlet berdasarkan tinggi dan berat
X = data[['height', 'weight']].dropna()

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

X['cluster'] = clusters

plt.scatter(X['height'], X['weight'], c=X['cluster'])
plt.xlabel("Tinggi Atlet (cm)")
plt.ylabel("Berat Atlet (kg)")
plt.title("Clustering Atlet berdasarkan Tinggi dan Berat")
plt.show()

#diagram bar jumlah atlet laki-laki dan perempuan
jumlah_sex = data['sex'].value_counts().plot.bar()
plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah Atlet")
plt.title("Jumlah Atlet Laki-laki dan Perempuan")
plt.show()