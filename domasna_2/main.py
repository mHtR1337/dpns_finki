import cv2
import numpy as np

def compass_operator(image, threshold=100):
    #филтри за compass операторот
    north_filter = np.array([[5, 5, 5],
                             [-3, 0, -3],
                             [-3, -3, -3]])
    south_filter = np.array([[-3, -3, -3],
                             [-3, 0, -3],
                             [5, 5, 5]])
    east_filter = np.array([[-3, -3, 5],
                            [-3, 0, 5],
                            [-3, -3, 5]])
    west_filter = np.array([[5, -3, -3],
                            [5, 0, -3],
                            [5, -3, -3]])

    #примена на филтрите на сликата
    north_edge = cv2.filter2D(image, -1, north_filter)
    south_edge = cv2.filter2D(image, -1, south_filter)
    east_edge = cv2.filter2D(image, -1, east_filter)
    west_edge = cv2.filter2D(image, -1, west_filter)

    #комбинација на резултатите од сите филтри
    combined_edges = cv2.max(north_edge, south_edge)
    combined_edges = cv2.max(combined_edges, east_edge)
    combined_edges = cv2.max(combined_edges, west_edge)

    #примена на праг за филтрирање на резултатите
    _, edges_thresh = cv2.threshold(combined_edges, threshold, 255, cv2.THRESH_BINARY)

    return edges_thresh

#читање на сликата
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#пресметување на compass операторот
edges = compass_operator(image, threshold=100)

#прикажување на резултатите
cv2.imshow('originalna slika', image)
cv2.imshow('compass slika', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()