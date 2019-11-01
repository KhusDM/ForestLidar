import laspy
import math
import pandas as pd

filename = 'areas/den.las'
inFile = laspy.file.File(filename, mode='r')
inHeader = laspy.header.HeaderManager(inFile.header, inFile.reader)
x_offset, y_offset, z_offset = inHeader.offset
x_scale, y_scale, z_scale = inHeader.scale

points_count = len(inFile.points)
print("Кол-во точек в файле: {0}".format(points_count))

avg_area_size = 1000000
areas_count = math.ceil(points_count / avg_area_size)
x_min, x_max = min(inFile.X), max(inFile.X)
x_abs = x_max - x_min
x_step = math.ceil(x_abs / areas_count)
print(x_min, x_max, areas_count, x_abs, x_step)

df = pd.DataFrame({
    "X": inFile.X,
    "Y": inFile.Y,
    "Z": inFile.Z,
    "Blue": inFile.Blue,
    "Green": inFile.Green,
    "Red": inFile.Red
}).sort_values(by=["X"])
# print(df)

x0 = x_min
x1 = x0 + x_step
for i in range(areas_count):
    area_df = df[(df.X >= x0) & (df.X < x1)]
    outfile = laspy.file.File("dendrarium_areas/d_area_{0}.las".format(i + 1), mode="w", header=inFile.header)
    outfile.X = area_df["X"].tolist()
    outfile.Y = area_df["Y"].tolist()
    outfile.Z = area_df["Z"].tolist()
    outfile.Blue = area_df["Blue"].tolist()
    outfile.Green = area_df["Green"].tolist()
    outfile.Red = area_df["Red"].tolist()
    outfile.close()

    x0 = x1
    x1 += x_step
