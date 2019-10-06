import laspy
import json
import plotly
# import plotly.plotly as py
import plotly.graph_objs as go
# from sklearn.cluster import KMeans
import numpy as np


def createLas(x, y, areaname, skip_step=1):
    pointsLas = []
    index = counter = 0
    while (index + skip_step <= points_count):
        x_point = inFile.X[index] * x_scale + x_offset
        y_point = inFile.Y[index] * y_scale + y_offset
        z_point = inFile.Z[index] * z_scale + z_offset

        if (x[0] <= x_point <= x[1] and y[0] <= y_point <= y[1] and 10 < z_point):
            pointsLas.append([inFile.X[index], inFile.Y[index], inFile.Z[index]])

        index += skip_step
        counter += 1

    points_area_count = len(pointsLas)
    print("Кол-во точек для {0}: {1}".format(areaname, points_area_count))

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


def mark_las(areas, a_x, b_x, a_y, b_y, skip_step=1):
    for area in areas:
        area['points'] = np.array(area['points'])
        area['points'][:, 0] = a_x * area['points'][:, 0] + b_x
        area['points'][:, 1] = a_y * area['points'][:, 1] + b_y

    index = 0
    points_by_areas = dict()
    while (index + skip_step <= points_count):
        x_point = inFile.X[index] * x_scale + x_offset
        y_point = inFile.Y[index] * y_scale + y_offset
        z_point = inFile.Z[index] * z_scale + z_offset

        isArea = False
        for i in range(0, len(areas)):
            points = areas[i]['points']
            x_min, x_max = min(points[:, 0]), max(points[:, 0])
            y_min, y_max = min(points[:, 1]), max(points[:, 1])

            if (x_min <= x_point <= x_max and y_min <= y_point <= y_max and 10 < z_point):
                if (not ('tree_{0}'.format(i + 1) in points_by_areas)):
                    points_by_areas['tree_{0}'.format(i + 1)] = ''

                r = g = b = 200
                points_by_areas['tree_{0}'.format(i + 1)] += '{0} {1} {2} {3} {4} {5}\n'.format(x_point, y_point,
                                                                                                z_point, r, g, b)
                isArea = True
                break

        if (not (isArea)):
            if (not ('other_1' in points_by_areas)):
                points_by_areas['other_1'] = ''

            r = g = b = 100
            points_by_areas['other_1'] += '{0} {1} {2} {3} {4} {5}\n'.format(x_point, y_point, z_point, r, g, b)

        if (not ('forest_1' in points_by_areas)):
            points_by_areas['forest_1'] = ''

        points_by_areas['forest_1'] += '{0} {1} {2} {3} {4} {5}\n'.format(x_point, y_point, z_point, r, g, b)
        index += skip_step

    for key in points_by_areas.keys():
        with open('label_files/{0}.txt'.format(key), 'a') as file:
            file.write(points_by_areas[key])


filename = 'areas/area1.las'
inFile = laspy.file.File(filename, mode='r')
inHeader = laspy.header.HeaderManager(inFile.header, inFile.reader)
x_offset, y_offset, z_offset = inHeader.offset
x_scale, y_scale, z_scale = inHeader.scale
header = laspy.header.Header()

points_count = len(inFile.points)
print("Кол-во точек в файле: {0}".format(points_count))

# x = [4207281, 4207335]
# y = [7544002, 7544041]
# h = 1
# createLas(x=x, y=y, skip_step=h, areaname="area1")
#
# x2 = [4207281, 4207323]
# y2 = [7543966, 7544000]
# h2 = 1
# createLas(x=x2, y=y2, skip_step=h2, areaname="area2")

with open('label_json/Box/Area1.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# area1.las coefficients
a_x = 0.06
b_x = 4207282.48
a_y = -0.06
b_y = 7544038.32

# area2.las coefficients
# a_x = 0.06
# b_x = 4207282.4
# a_y = -0.052
# b_y = 7543998.9

# h = 1
# i = 1
# for item in data['shapes']:
#     points = np.array(item['points'])
#     x_min, x_max = min(points[:, 0]), max(points[:, 0])
#     y_min, y_max = min(points[:, 1]), max(points[:, 1])
#     x = [x_min * a_x + b_x, x_max * a_x + b_x]
#     y = [y_max * a_y + b_y, y_min * a_y + b_y]
#     createLas(x=x, y=y, skip_step=h, areaname="area1_tree{0}".format(i))
#     i += 1

h = 100
mark_las(data['shapes'], a_x, b_x, a_y, b_y, skip_step=h)
