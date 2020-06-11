import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ioValues():
    archive = open('dataset_n2020_rev_final.txt', 'r')
    lines = archive.readlines()
    archive.close()

    # input
    distance = []
    time = []
    volume = []
    weight = []
    input_values = []

    # output
    contaminated = []

    range_lines = 0
    for line in lines:
        line_values = line.split()

        distance.append(float(line_values[0]))
        time.append(float(line_values[1]))
        volume.append(float(line_values[2]))
        weight.append(float(line_values[3]))
        contaminated.append(int(line_values[4]))

        range_lines += 1

    for i in range(range_lines):
        input_values.append([
            distance[i],
            time[i],
            volume[i],
            weight[i]
        ])

    return input_values, contaminated


def newData():
    distance = [5.0052, 5.58135, 5.82679, 6.33638, 5.31261, 5.14922, 7.22062, 5.98769, 5.38889, 6.35219]
    time = [2.27383, 3.87798, 2.86186, 5.38318, 3.47496, 3.90657, 3.36719, 4.56878, 4.97743, 4.44608]
    volume = [4.65391, 3.8874, 6.8836, 6.0182, 5.39429, 4.63587, 7.32428, 5.29755, 6.13952, 5.53578]
    weight = [5.94092, 3.80204, 3.96422, 1.42675, 5.38155, 5.11076, 2.2763, 3.44059, 2.19698, 2.30308]
    input_values = []
    for i in range(10):
        input_values.append([
            distance[i],
            time[i],
            volume[i],
            weight[i]
        ])
    return input_values


def plotGraph2D(input_values_2d, contaminated, classifier, graph_title):
    x_min, x_max = input_values_2d[:, 0].min() - 1, input_values_2d[:, 0].max() + 1
    y_min, y_max = input_values_2d[:, 1].min() - 1, input_values_2d[:, 1].max() + 1
    xx, yy, = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title(graph_title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.scatter(input_values_2d[:, 0], input_values_2d[:, 1], c=contaminated, cmap=plt.cm.coolwarm)
    plt.savefig('graphs/' + graph_title + '.png')
    # plt.show()


def plotGraph3D(input_values_3d, contaminated, classifier, graph_title):
    x_min, x_max = input_values_3d[:, 0].min(), input_values_3d[:, 0].max()
    y_min, y_max = input_values_3d[:, 1].min(), input_values_3d[:, 1].max()
    z_min, z_max = input_values_3d[:, 2].min(), input_values_3d[:, 2].max()
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1), np.arange(z_min, z_max, 1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_values_3d[:, 0], input_values_3d[:, 1], input_values_3d[:, 2], c=contaminated)
    ax.plot_trisurf(input_values_3d[:, 0], input_values_3d[:, 1], input_values_3d[:, 2], linewidth=0, antialiased=False,
                    color='gray', alpha='0.1')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC2')
    plt.title(graph_title)
    plt.savefig('graphs/' + graph_title + '.png')
    # plt.show()

# with open('output_data/predict_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Distancia", "Time", "Volume", "Weight"])
