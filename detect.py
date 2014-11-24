import numpy as np
import cv2

FILE_NAME = "2.jpg"

def order_points(points):
    # Returns a list of points such that: [top_left, top_right, bottom_right, bottom_left]
    rect = np.zeros((4, 2), dtype = "float32")

    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points):
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    # compute the width of the new image
    widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[0] - bottom_left[0]) ** 2))
    widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[0] - top_left[0]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.sqrt(((top_right[1] - bottom_right[1]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    heightB = np.sqrt(((top_left[1] - bottom_left[1]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def get_largest_contour(contours, number_corners, max_number_contours = 300):
    # Returns the corners of the largest contour with the nummber of corners
    bestHull = None
    bestHullArea = 0
    for contour in contours:
        hull = cv2.convexHull(contour)
        hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
        area = cv2.contourArea(hull)
        if len(hull) == number_corners and area > bestHullArea:
            bestHull = hull
            bestHullArea = area

    return bestHull


ratio = 0.5
for i in xrange(1, 15):
    image = cv2.imread("input/" + str(i) + ".jpg")
    original = image.copy()

    # Resize the image / convert to grayscale / apply a blur
    image = cv2.resize(image, (0, 0), fx = ratio, fy = ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 75, 200)

    # Contour and corner detection
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = get_largest_contour(contours, 4)
    cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)

    # Transformation
    warped = four_point_transform(original, hull.reshape(4, 2) / ratio)

    # Display images
    cv2.imwrite("output/" + str(i) + "_edges.jpg", image)
    cv2.imwrite("output/" + str(i) + "_warped.jpg", warped)
    # cv2.imshow("Contour", image)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Warped", warped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
