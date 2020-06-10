import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ia_n2020


def normalize_data_2d():
    input_values, contaminated = ia_n2020.ioValues()
    predict_values = ia_n2020.newData()

    norm = StandardScaler()
    input_values_norm = norm.fit_transform(input_values)
    predict_values_norm = norm.fit_transform(predict_values)
    pca = PCA(n_components=2)
    input_values_2d = pca.fit_transform(input_values_norm)
    predict_values_2d = pca.fit_transform(predict_values_norm)

    reg1 = ''
    reg2 = ''
    for i in range(len(input_values_2d)):
        if contaminated[i] == 0:
            reg1 = plt.scatter(input_values_2d[i][0], input_values_2d[i][1], marker='x', color='g')
        elif contaminated[i] == 1:
            reg2 = plt.scatter(input_values_2d[i][0], input_values_2d[i][1], marker='o', color='b')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.legend((reg1, reg2), ('Não contaminado', 'Contaminado'))
    plt.savefig('graphs/pca_graph.png')
    plt.close()

    return input_values_2d, predict_values_2d


def normalize_data_3d():
    input_values, contaminated = ia_n2020.ioValues()
    predict_values = ia_n2020.newData()

    norm = StandardScaler()
    input_values_norm = norm.fit_transform(input_values)
    predict_values_norm = norm.fit_transform(predict_values)
    pca = PCA(n_components=3)
    input_values_3d = pca.fit_transform(input_values_norm)
    predict_values_3d = pca.fit_transform(predict_values_norm)

    return input_values_3d, predict_values_3d
