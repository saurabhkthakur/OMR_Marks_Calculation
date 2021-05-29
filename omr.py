import cv2
import transform
import imutils
import numpy as np
from imutils import contours

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread('omr.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
median= np.median(image)
print(int(max(0, (1-0.33) * median)))
print(int(min(255, (1+0.33) * median)))
edged = cv2.Canny(blur, 40, 135)

cv2.imshow('edged', edged)
cv2.waitKey(0)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
doCnt = None

if len(cnts)>0:

    cnts= sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c,0.02*peri, True)

        if len(approx) == 4:
            doCnt = approx
            break

cv2.drawContours(image, [approx], -1, (0,0,255),2)
cv2.imshow('outline',image)
cv2.waitKey(0)


paper = transform.four_point_transform(image, doCnt.reshape(4,2))
warped = transform.four_point_transform(gray, doCnt.reshape(4,2))

cv2.imshow('paper', paper)
cv2.imshow('warped', warped)
cv2.waitKey(0)

thresh = cv2.threshold(warped, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow('thresh', thresh)
cv2.waitKey(0)




cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:

    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)

    if w>=20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)




questionCnts = contours.sort_contours(questionCnts, method='top-to-bottom')[0]
correct = 0

for (q,i) in enumerate(np.arange(0,len(questionCnts), 5)):

    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):

        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        cv2.imshow('maskedddd', mask)
        cv2.waitKey(0)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    cv2.imshow('pappp', paper)
    cv2.waitKey(0)

print(len(cnts))
print(correct)
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)





