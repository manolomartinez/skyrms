import numpy as np

# Takes a matrix with a probability vector and a value in each row and draws a simplex plot

def coord_conversion(triple): # Takes a vector to a point in the simplex
    a = triple[0]
    b = triple[1]
    c = triple[2]
    return (1/2*(2*b+c)/(a+b+c), np.sqrt(3)/2*c/(a+b+c))

def input_to_plot(matrix):
    points = []
    heights = []
    for row in matrix:
        points.append([coord_conversion(row[0]), row[1]])
    return points

def simplex_plot(matrix):
    triangle = polygon([coord_conversion(n) for n in [[1,0,0],[0,1,0],[0,0,1]]], fill = False)
    points = input_to_plot(matrix)
    pointsplot = sum(point(i[0], size=(i[1]*100 + 10), faceted = True) for i in points)
    plot =  pointsplot + axes
    return plot.save("triage.png", axes = False)
   
   



