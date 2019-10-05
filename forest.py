import laspy
import json
import plotly
# import plotly.plotly as py
import plotly.graph_objs as go
# from sklearn.cluster import KMeans
import numpy as np


def createLas(x, y, areaname, skip_step=1):
    pointsLas = []
    # pointsPlot = []
    index = counter = 0
    while (index + skip_step <= points_count):
        # points.append(
        #     [inFile.X[index] * x_scale + x_offset, inFile.Y[index] * y_scale + y_offset,
        #      (-1) * inFile.Z[index] * z_scale + z_offset])
        x_point = inFile.X[index] * x_scale + x_offset
        y_point = inFile.Y[index] * y_scale + y_offset
        z_point = inFile.Z[index] * z_scale + z_offset

        if (x[0] <= x_point <= x[1] and y[0] <= y_point <= y[1]):
            # pointsPlot.append([x_point, y_point, z_point])
            pointsLas.append([inFile.X[index], inFile.Y[index], inFile.Z[index]])

        index += skip_step
        counter += 1

    points_area_count = len(pointsLas)
    print("Кол-во точек для {0}: {1}".format(areaname, points_area_count))

    # pointsPlot = np.array(pointsPlot)
    # kmeans = KMeans(n_clusters=4, random_state=0).fit(pointsPlot)
    # labels = kmeans.labels_

    # plotly.offline.plot({
    #     "data": [go.Scatter3d(x=pointsPlot[:, 0], y=pointsPlot[:, 1], z=pointsPlot[:, 2], showlegend=False,
    #                           mode='markers',
    #                           marker=dict(size=1, color='green', line=dict(color='black', width=1)))],
    #     "layout": go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    # }, filename="{0}_plot.html".format(areaname))

    pointsLas = np.array(pointsLas)
    outfile = laspy.file.File("tree_clouds\\{0}.las".format(areaname), mode="w", header=inFile.header)
    outfile.X = pointsLas[:, 0]
    outfile.Y = pointsLas[:, 1]
    outfile.Z = pointsLas[:, 2]
    outfile.close()


filename = 'area1.las'
inFile = laspy.file.File(filename, mode='r')
inHeader = laspy.header.HeaderManager(inFile.header, inFile.reader)
x_offset, y_offset, z_offset = inHeader.offset
x_scale, y_scale, z_scale = inHeader.scale
header = laspy.header.Header()

points_count = len(inFile.points)
print("Кол-во точек в файле: {0}".format(points_count))
#
# x = [4207281, 4207335]
# y = [7544002, 7544041]
# h = 1
# createLas(x=x, y=y, skip_step=h, areaname="area1")
#
# x2 = [4207281, 4207323]
# y2 = [7543966, 7544000]
# h2 = 1
# createLas(x=x2, y=y2, skip_step=h2, areaname="area2")

with open('label_json/Box/Area1.json', 'r', encoding='utf-8') as fh:
    data = data = json.load(fh)

a_x = 0.06
b_x = 4207282.48
a_y = -0.06
b_y = 7544038.32
i = 1
for item in data['shapes']:
    points = item['points']
    x = [points[0][0] * a_x + b_x, points[2][0] * a_x + b_x]
    y = [points[2][1] * a_y + b_y, points[0][1] * a_y + b_y]
    h = 1
    createLas(x=x, y=y, skip_step=h, areaname="area1_tree{0}".format(i))
    i += 1

# trace = go.Scatter3d(x=x_axis, y=y_axis, z=z_axis, showlegend=False, mode='markers',
#                      marker=dict(size=1, color=['#FF0000', '#00FF00', '#0000FF', '#FF0000'],
#                                  line=dict(color='black', width=1)))

# fig = go.Figure(data=[go.Scatter3d(x=pointsPlot[:, 0], y=pointsPlot[:, 1], z=pointsPlot[:, 2],
#                                    mode='markers')])
# fig.show()

# trace = go.Scatter3d(x=x_axis, y=y_axis, z=z_axis, mode='markers', marker=dict(
#     size=12,
#     line=dict(
#         color='rgba(217, 217, 217, 0.14)',
#         width=0.5
#     ),
#     opacity=0.8
# ))
#
# data = [trace]
# layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
# )
# fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='test', fileopt='overwrite')

# np.random.seed(5)
#
# fig = plotly.tools.make_subplots(rows=2, cols=3,
#                                  print_grid=False,
#                                  specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
#                                         [{'is_3d': True, 'rowspan': 1}, None, None]])
# scene = dict(
#     camera=dict(
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=2.5, y=0.1, z=0.1)
#     ),
#     xaxis=dict(
#         range=[-1, 4],
#         title='Petal width',
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)',
#         showticklabels=False, ticks=''
#     ),
#     yaxis=dict(
#         range=[4, 8],
#         title='Sepal length',
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)',
#         showticklabels=False, ticks=''
#     ),
#     zaxis=dict(
#         range=[1, 8],
#         title='Petal length',
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)',
#         showticklabels=False, ticks=''
#     )
# )

# centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# estimators = {'k_means_iris_3': KMeans(n_clusters=3),
#               'k_means_iris_8': KMeans(n_clusters=8),
#               'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
#                                               init='random')}
# fignum = 1
# for name, est in estimators.items():
#     est.fit(X)
#     labels = est.labels_
#
#     trace = go.Scatter3d(x=X[:, 3], y=X[:, 0], z=X[:, 2],
#                          showlegend=False,
#                          mode='markers',
#                          marker=dict(
#                              color=labels.astype(np.float),
#                              line=dict(color='black', width=1)
#                          ))
#     fig.append_trace(trace, 1, fignum)
#
#     fignum = fignum + 1

# y = np.choose(y, [1, 2, 0]).astype(np.float)
#
# trace1 = go.Scatter3d(x=X[:, 3], y=X[:, 0], z=X[:, 2],
#                       showlegend=False,
#                       mode='markers',
#                       marker=dict(
#                           color=y,
#                           line=dict(color='black', width=1)))
# fig.append_trace(trace1, 2, 1)
#
# fig['layout'].update(height=900, width=900,
#                      margin=dict(l=10, r=10))
#
# fig['layout']['scene1'].update(scene)
# fig['layout']['scene2'].update(scene)
# fig['layout']['scene3'].update(scene)
# fig['layout']['scene4'].update(scene)
# fig['layout']['scene5'].update(scene)
#
# py.plot(fig)

# plotly.tools.set_credentials_file(username='KhusDM', api_key='Z1Aj6ViW6P63pXVC4drS')
# trace1 = go.Scatter3d(
#     x=x_axis,
#     y=y_axis,
#     z=z_axis,
#     mode='markers',
#     marker=dict(
#         size=2,
#         line=dict(
#             color='rgba(217, 217, 217, 0.14)',
#             width=0.5
#         ),
#         opacity=0.8
#     )
# )
#
# data = [trace1]
# layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
# )
# fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='simple-3d-scatter', fileopt='overwrite')

# kmeans = KMeans(n_clusters=4, random_state=0).fit(points)

# x_axis = []
# y_axis = []
# z_axis = []
# for point in points:
#     x_axis.append(point[0])
#     y_axis.append(point[1])
#     z_axis.append((-1) * point[2])

# fig = plt.figure()
# ax = Axes3D(fig)
# # ax = fig.add_subplot(111, projection='3d')
# plt.scatter(x_axis, y_axis, z_axis, zdir="z")
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim(min(x_axis), max(x_axis))
# ax.set_ylim(min(y_axis), max(y_axis))
# ax.set_zlim(min(z_axis), max(z_axis))
# # ax.set_xlim(min(inFile.X), max(inFile.X))
# # ax.set_ylim(min(inFile.Y), max(inFile.Y))
# # ax.set_zlim(min(inFile.Z), max(inFile.Z))
# # ax.set_xlim(min(inFile.X) + k, max(inFile.X) + k)
# # ax.set_ylim(min(inFile.Y) + k, max(inFile.Y) + k)
# # ax.set_zlim(min(inFile.Z) + k, max(inFile.Z) + k)
# plt.show()

# # Определяем модель и скорость обучения
# model = TSNE(n_components=3, learning_rate=100)
#
# # Обучаем модель
# transformed = model.fit_transform(points)
#
# # Представляем результат в двумерных координатах
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]
# z_axis = transformed[:, 2]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.scatter(x_axis, y_axis, z_axis)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
