import cv2
import transform
import imutils
import numpy as np
from imutils import contours

image = cv2.imread('2.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3),0)
median= np.median(image)
print(int(max(0, (1-0.33) * median)))
print(int(min(255, (1+0.33) * median)))

edged = cv2.Canny(blur, 150,250)
cv2.imshow('edged', edged)
cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnts= sorted(cnts, key=cv2.contourArea, reverse=True)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doCnt = approx
        break
cv2.drawContours(image, [doCnt], -1, (0,0,255),2)
cv2.imshow('outlined', image)
cv2.waitKey(0)


paper = transform.four_point_transform(image, doCnt.reshape(4,2))
warped = transform.four_point_transform(gray, doCnt.reshape(4,2))

cv2.imshow('paper', paper)
cv2.imshow('warped', warped)
cv2.waitKey(0)

thresh = cv2.threshold(warped,  0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
cnts = imutils.grab_contours(cnts)
qCnts = []

for c in cnts:

    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/ float(h)

    if w>=10 and h>=10 and ar>= 0.9 and ar <=1.1:
        qCnts.append(c)



qCnts= contours.sort_contours(qCnts, method='top-to-bottom')[0]
correct = 0

for c in qCnts:
    cv2.drawContours(paper, [c], -1, (0,0,255), 2)
    cv2.imshow('paper',paper)
    cv2.waitKey(0)