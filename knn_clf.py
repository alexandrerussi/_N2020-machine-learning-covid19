from sklearn.neighbors import KNeighborsClassifier
import ia_n2020
import pca_graph
import pandas as pd

input_values, contaminated = ia_n2020.ioValues()
predict_values = ia_n2020.newData()
input_values_2d, predict_values_2d = pca_graph.normalize_data_2d()

# knn classifier with 4 input data
knn_clf = KNeighborsClassifier(n_neighbors=4)
knn_clf.fit(input_values, contaminated)

knn_score = knn_clf.score(input_values, contaminated)
print(knn_score)

knn_predict = knn_clf.predict(predict_values)
print(knn_predict)

spreadsheet = pd.read_csv('output_data/predict_data.csv')
spreadsheet['knn_predict'] = knn_predict
spreadsheet.to_csv('output_data/predict_data.csv', index=False)

# knn classifier with 2D data
knn_clf_2d = KNeighborsClassifier(n_neighbors=4)
knn_clf_2d.fit(input_values_2d, contaminated)

knn_score_2d = knn_clf_2d.score(input_values_2d, contaminated)
print(knn_score_2d)

knn_predict_2d = knn_clf_2d.predict(predict_values_2d)
print(knn_predict_2d)

ia_n2020.plotGraph2D(input_values_2d, contaminated, knn_clf_2d, 'graph_knn_2d')
