import cv2
import numpy as np
import math

answers = {0: "A", 1: "B", 2: "C", 3: "D"}
model_answers = {1: "B", 2: "C", 3: "A", 4: "A", 5: "D", 6: "A", 7: "C", 8: "C", 9: "A", 10: "C",
                 11: "A", 12: "B", 13: "C", 14: "C", 15: "B", 16: "A", 17: "D", 18: "B", 19: "C", 20: "B",
                 21: "D", 22: "C", 23: "D", 24: "B", 25: "D", 26: "C", 27: "D", 28: "D", 29: "B", 30: "C",
                 31: "B", 32: "B", 33: "D", 34: "C", 35: "B", 36: "C", 37: "B", 38: "C", 39: "C", 40: "A",
                 41: "B", 42: "B", 43: "C", 44: "C", 45: "B"}
total_grade = 0
original_image = cv2.imread("/home/meegz/Projects/Image Processing/Dataset/test/S_2_hppscan19.png")
original_image = original_image[670: 1480, :]
height, width = original_image.shape[:2]
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (7, 7), 0.5)
edge_image = cv2.Canny(blurred, 60, 100, L2gradient=True)
LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines, w, prec, nfa = LSD.detect(blurred)
v_threshold_length = 730
v_threshold_angle = range(70, 110)
v_filter = []
h_threshold_length = 200
h_threshold_angle_1 = range(160, 180)
h_threshold_angle_2 = range(0, 20)
h_filter = []
for line in lines:
    for x1, y1, x2, y2 in line:
        if y1 < y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        theta = cv2.fastAtan2((y2 - y1), (x2 - x1))
        if theta > 180:
            theta -= 180
        length = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        if int(theta) in v_threshold_angle and length >= v_threshold_length and (int(x1) in range(80, 200) or int(x1) in range(1040, 1180)):
            v_filter.append([x1, y1, x2, y2])
        if (int(theta) in h_threshold_angle_1 or int(theta) in h_threshold_angle_2) and length >= h_threshold_length:
            h_filter.append([x1, y1, x2, y2])
pts = []
for v_line in v_filter:
    vx1, vy1, vx2, vy2 = v_line[0], v_line[1], v_line[2], v_line[3]
    for h_line in h_filter:
        hx1, hy1, hx2, hy2 = h_line[0], h_line[1], h_line[2], h_line[3]
        if abs(vx1 - hx1) < 10:
            pts.append([vx1, hy1, 0])
        elif abs(vx1 - hx2) < 10:
            pts.append([vx1, hy2, 0])
        elif abs(vx2 - hx2) < 10:
            pts.append([vx2, hy2, 0])
        elif abs(vx2 - hx1) < 10:
            pts.append([vx2, hy1, 0])
indexes = []
for i in range(len(pts)):
    x1, y1 = pts[i][0], pts[i][1]
    pts[i][2] = 1
    for j in range(len(pts)):
        if pts[j] == pts[i] or pts[j][2] == 0:
            continue
        x2, y2 = pts[j][0], pts[j][1]
        if abs(x1 - x2) < 5 and abs(y1 - y2) < 5:
            indexes.append(j)
indexes = sorted(indexes, reverse=True)
for i in indexes:
    pts.remove(pts[i])
filtered_pts = []
for pt in pts:
    filtered_pts.append([pt[0], pt[1]])
assert len(filtered_pts) == 4
dst = np.array([[0, 0], [height, 0], [height, width], [0, width]], dtype="float32")
filtered_pts = np.array(filtered_pts, dtype="float32")
rect = np.zeros((4, 2), dtype="float32")
s = filtered_pts.sum(axis=1)
rect[0] = filtered_pts[np.argmin(s)]
rect[2] = filtered_pts[np.argmax(s)]
diff = np.diff(filtered_pts, axis=1)
rect[1] = filtered_pts[np.argmin(diff)]
rect[3] = filtered_pts[np.argmax(diff)]
HomographyToInv = cv2.getPerspectiveTransform(rect, dst)
new_image = cv2.warpPerspective(original_image, HomographyToInv, (height, width))
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
# Resized 3shan kasselt a5od dimensions gdeda :v *Magdy*
new_image = cv2.resize(new_image, (1083, 1240), interpolation=cv2.INTER_AREA)
col1 = new_image[:, 0:331]
col2 = new_image[:, 350:691]
col3 = new_image[:, 710:]
questions = []
col1l = []
col2l = []
col3l = []
offset = 68
for i in range(0, 15):
    col1l.append(col1[115 + i*offset: 115 + (i+1)*offset])
    col2l.append(col2[115 + i*offset: 115 + (i+1)*offset])
    col3l.append(col3[115 + i*offset: 115 + (i+1)*offset])
questions.append(col1l)
questions.append(col2l)
questions.append(col3l)
ans = np.zeros((4, 2))
question_number_offset = 0
for k in range(0, 3):
    for j in range(0, 15):
        edges_question = cv2.Canny(questions[k][j], 60, 70, L2gradient=True)
        im3, cntss, hirc = cv2.findContours(edges_question, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntss_sorted_circles = sorted(cntss, key=cv2.contourArea, reverse=True)
        questions[k][j] = cv2.cvtColor(questions[k][j], cv2.COLOR_GRAY2BGR)
        cv2.imshow('fsad', questions[k][j])
        cv2.waitKey(0)
        ans = []
        _range = 3
        i = -1
        while i < _range:
            i += 1
            x, y, w, h = cv2.boundingRect(cntss_sorted_circles[i])
            if x < 110:
                _range += 1
                continue
            cv2.rectangle(questions[k][j], (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.drawContours(questions[k][j], cntss_sorted_circles, i, (0, 0, 255), 2, cv2.LINE_AA)
            rect = questions[k][j]
            rect = rect[y:y + h, x:x + w]
            mean = rect.mean()
            ans.append([x, mean])
        sorted_ans = sorted(ans, key=lambda l: l[0], reverse=False)
        final_answer = []
        for i in range(0, len(ans)):
            if sorted_ans[i][1] < 170:
                final_answer.append(answers[i])
        if len(final_answer) == 1:
            if final_answer[0] == model_answers[k + 1 + question_number_offset + j]:
                total_grade += 1
        print("Question", (k + 1 + question_number_offset + j), ":", final_answer)
    question_number_offset += 14
print("Total Grade:", total_grade)
# while (cv2.waitKey(0) & 0xFF) != ord('q'):
#     pass
# cv2.destroyAllWindows()
