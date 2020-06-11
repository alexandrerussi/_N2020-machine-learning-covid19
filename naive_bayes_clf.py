from sklearn.naive_bayes import GaussianNB
import pca_graph
import ia_n2020

input_values, contaminated = ia_n2020.ioValues()
input_values_2d, predict_values_2d = pca_graph.normalize_data_2d()

nb_clf = GaussianNB()
nb_clf.fit(input_values_2d, contaminated)
nb_score = nb_clf.score(input_values_2d, contaminated)
nb_predict = nb_clf.predict(predict_values_2d)

print(nb_score)
print(nb_predict)

ia_n2020.plotGraph2D(input_values_2d, contaminated, nb_clf, 'naive_bayes_2d')
