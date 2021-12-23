import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialization
cap = cv2.VideoCapture('Videos/vid (6).mp4')

# Create color finder object
myColorFinder = ColorFinder(False) # True as input => find ball color in debug mode | False => finding the ball
hsvVals = {'hmin': 3, 'smin': 112, 'vmin': 0, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Position list
posList = []
xList = [item for item in range(0, 1300)]

# Target location
y_target = 590 # y-axis coordinate of the rim
x1_rim = 330
x2_rim = 430
prediction = False

while True:
    # Grab the image
    ret, img = cap.read()
    # img = cv2.imread("Ball.png") # use this to get the color
    img = img[0:900, :]

    # Find the color of the ball
    imgColor, mask = myColorFinder.update(img, hsvVals)
    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=200)

    if contours:
        posList.append(contours[0]['center']) # sorted in ascending order

    if posList:
        # Display the centers
        for i, pos in enumerate(posList):
            cv2.circle(imgContours, pos, 7, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 2)
            else:
                cv2.line(imgContours, pos, posList[i - 1], (0, 255, 0), 2)

        # Polynomial Regression y = Ax^2 +Bx + C
        # Find the coefficients
        coords = list(map(list, zip(*posList))) # unzip the list to 2 lists of coordinates zip(*iterable) to unzip
        A, B, C = np.polyfit(coords[0], coords[1], 2) # second order polynomial function

        for x in xList:
            y = int(A*x**2 + B*x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        # Prediction
        # X values 330 to 430 for Y = 590 => BASKET !
        if len(posList)<10:
            x_val = np.roots([A, B, C-y_target]) # we use the second solution

        prediction = x1_rim < x_val[1] < x2_rim
        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 100), thickness=5, colorR=(0, 225,0))
        else:
            cvzone.putTextRect(imgContours, "No basket", (50, 100), thickness=5, colorR=(0, 0, 225))


    # Display
    # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.5, 0.5)

    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(100)
