import laspy
import numpy as np

filename = 'areas/den.las'
inFile = laspy.file.File(filename, mode='r')
inHeader = laspy.header.HeaderManager(inFile.header, inFile.reader)
x_offset, y_offset, z_offset = inHeader.offset
x_scale, y_scale, z_scale = inHeader.scale

points_count = len(inFile.points)
print("Кол-во точек в файле: {0}".format(points_count))

area_size = 1000000
pointsLas = []
area_counter = 1
for i in range(points_count):
    pointsLas.append([inFile.X[i], inFile.Y[i], inFile.Z[i]])
    if ((i + 1) % area_size == 0 or i == points_count - 1):
        pointsLas = np.array(pointsLas)
        outfile = laspy.file.File("dendrarium_areas\\d_area_{0}.las".format(area_counter), mode="w",
                                  header=inFile.header)
        outfile.X = pointsLas[:, 0]
        outfile.Y = pointsLas[:, 1]
        outfile.Z = pointsLas[:, 2]
        outfile.close()

        area_counter += 1
        pointsLas = []
