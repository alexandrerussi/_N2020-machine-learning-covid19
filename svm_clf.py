from sklearn import svm
import pandas as pd
import ia_n2020
import pca_graph

input_values, contaminated = ia_n2020.ioValues()
predict_values = ia_n2020.newData()
input_values_2d, predict_values_2d = pca_graph.normalize_data_2d()
input_values_3d, predict_values_3d = pca_graph.normalize_data_3d()

# svm classifier with 4 input data
# sv_clf = svm.SVC(kernel='linear', C=0.5)
sv_clf = svm.SVC(kernel='rbf', gamma=1, C=0.5)
# sv_clf = svm.SVC(kernel='poly', degree=3, C=0.5)
sv_clf.fit(input_values, contaminated)

sv_score = sv_clf.score(input_values, contaminated)
print(sv_score)

sv_predict = sv_clf.predict(predict_values)
print(sv_predict)

spreadsheet = pd.read_csv('output_data/predict_data.csv')
spreadsheet['svm_predict'] = sv_predict
spreadsheet.to_csv('output_data/predict_data.csv', index=False)

# svm classifier with 2D data
sv_clf_2d = svm.SVC(kernel='linear', C=0.5)
# sv_clf_2d = svm.SVC(kernel='rbf', gamma=1, C=0.5)
# sv_clf_2d = svm.SVC(kernel='poly', degree=3, C=0.5)
sv_clf_2d.fit(input_values_2d, contaminated)

sv_score_2d = sv_clf_2d.score(input_values_2d, contaminated)
print(sv_score_2d)

sv_predict_2d = sv_clf_2d.predict(predict_values_2d)
print(sv_predict_2d)

ia_n2020.plotGraph2D(input_values_2d, contaminated, sv_clf_2d, 'graph_svm_2d')

# svm classifier with 3D data
# sv_clf = svm.SVC(kernel='linear', C=0.5)
sv_clf_3d = svm.SVC(kernel='rbf', gamma=1, C=0.5)
# sv_clf = svm.SVC(kernel='poly', degree=3, C=0.5)
sv_clf_3d.fit(input_values_3d, contaminated)

sv_score_3d = sv_clf_3d.score(input_values_3d, contaminated)
print(sv_score_3d)

sv_predict_3d = sv_clf_3d.predict(predict_values_3d)
print(sv_predict_3d)

ia_n2020.plotGraph3D(input_values_3d, contaminated, sv_clf_3d, 'graph_svm_3d')
