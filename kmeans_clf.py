from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ia_n2020
import pca_graph
import pandas as pd

input_values_2d, predict_values_2d = pca_graph.normalize_data_2d()

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(input_values_2d)
    inertia.append(kmeans.inertia_)

plt.plot(inertia)
plt.xlabel('# Cluster - K')
plt.ylabel("Inertia")
plt.savefig('graphs/graph_kmeans_inertia')
plt.show()

kmeans_clf = KMeans(n_clusters=2)
kmeans_clf.fit(input_values_2d)
contaminated = kmeans_clf.predict(input_values_2d)
centroid = kmeans_clf.cluster_centers_

contaminated_predict = kmeans_clf.predict(predict_values_2d)
print(contaminated_predict)

spreadsheet = pd.read_csv('output_data/predict_data.csv')
spreadsheet['kmeans_predict'] = contaminated_predict
spreadsheet.to_csv('output_data/predict_data.csv', index=False)

plt.scatter(input_values_2d[:, 0], input_values_2d[:, 1], s=25, alpha=0.4, c=contaminated)
plt.title('graph_kmeans_2d')
plt.savefig('graphs/graph_kmeans_2d_v2')
plt.show()

ia_n2020.plotGraph2D(input_values_2d, contaminated, kmeans_clf, 'graph_kmeans_2d')
