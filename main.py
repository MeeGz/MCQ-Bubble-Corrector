import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


original_image = cv2.imread("/home/yousef/projects/mcq-corrector/dataset/test/S_1_hppscan1.png")
original_image = original_image[670:, :]
height, width = original_image.shape[:2]
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
edge_image = cv2.Canny(blurred, 60, 100, L2gradient=True)
im2, cnts, h = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
answer_block = cnts_sorted[0]
peri = cv2.arcLength(answer_block, True)
approx = cv2.approxPolyDP(answer_block, 0.02 * peri, True)
assert len(approx) == 4
new_approx = []
for pt in approx:
    for x, y in pt:
        new_approx.append([x, y])
dst = np.array([[0, 0], [height, 0], [height, width], [0, width]], dtype="float32")
pts = np.array(new_approx, dtype="float32")
rect = order_points(pts)
HomographyToInv = cv2.getPerspectiveTransform(rect, dst)
new_image = cv2.warpPerspective(original_image, HomographyToInv, (height, width))
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
col1 = new_image[:, 0:331]
col2 = new_image[:, 350:691]
col3 = new_image[:, 710:]
new_image = cv2.resize(new_image, (500, 500), interpolation=cv2.INTER_AREA)
questions = []
col1l = []
col2l = []
col3l = []
offset = 69
for i in range(0, 15):
    col1l.append(col1[110 + i*offset: 110 + (i+1)*offset])
    col2l.append(col2[110 + i*offset: 110 + (i+1)*offset])
    col3l.append(col3[110 + i*offset: 110 + (i+1)*offset])
questions.append(col1l)
questions.append(col2l)
questions.append(col3l)
ans = np.zeros((4, 2))
for k in range(0, 3):
    for j in range(0, 15):
        edges_question = cv2.Canny(questions[k][j], 60, 100, L2gradient=True)
        im3, cntss = cv2.findContours(edges_question, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntss_sorted_circles = sorted(cntss, key=cv2.contourArea, reverse=True)
        questions[k][j] = cv2.cvtColor(questions[k][j], cv2.COLOR_GRAY2BGR)
        for i in range(0, 4):
            x, y, w, h = cv2.boundingRect(cntss_sorted_circles[i])
            cv2.rectangle(questions[k][j], (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(questions[k][j], cntss_sorted_circles, i, (0, 0, 255), 2, cv2.LINE_AA)
            rect = questions[k][j]
            rect = rect[y:y + h, x:x + h]
            hh, ww = questions[k][j].shape[:2]
            mean = rect.mean()
            ans[i, :] = x, mean
        mask = ans[:, 1] < 150
        choosed_ans = ans[mask]
        print(choosed_ans)
cv2.imshow("hee", questions[0][10])
cv2.imshow('hi', new_image)
while (cv2.waitKey(0) & 0xFF) != ord('q'):
    pass
cv2.destroyAllWindows()
