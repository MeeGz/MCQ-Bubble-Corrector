import cv2
import numpy as np
import math
import csv
import os

answers = {0: "A", 1: "B", 2: "C", 3: "D"}
model_answers = {1: "B", 2: "C", 3: "A", 4: "A", 5: "D", 6: "A", 7: "C", 8: "C", 9: "A", 10: "C",
                 11: "A", 12: "B", 13: "C", 14: "C", 15: "B", 16: "A", 17: "D", 18: "B", 19: "C", 20: "B",
                 21: "D", 22: "C", 23: "D", 24: "B", 25: "D", 26: "C", 27: "D", 28: "D", 29: "B", 30: "C",
                 31: "B", 32: "B", 33: "D", 34: "C", 35: "B", 36: "C", 37: "B", 38: "C", 39: "C", 40: "A",
                 41: "B", 42: "B", 43: "C", 44: "C", 45: "B"}
total_grade = 0
faults = 0
dir_path = "/home/yousef/projects/mcq-corrector/dataset/correction/"
write_list = []
toWrite = []
wrong_detection_count = 0
DEBUG = False
fileCount = 0

for filename in os.listdir(dir_path):
    fileCount += 1
    print("------------------------------------------------")
    print("File:", filename)
    original_image = cv2.imread(dir_path + "/" + filename)
    original_image = original_image[650: 1580, :]
    hoppa = original_image.copy()
    height, width = original_image.shape[:2]
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0.5)
    LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, w, prec, nfa = LSD.detect(blurred)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(hoppa, (x1, y1), (x2, y2), (0, 255, 255), 1)
    v_threshold_length = 300
    v_threshold_angle = range(60, 120)
    v_filter = []
    h_threshold_length = 50
    h_threshold_angle_1 = range(150, 180)
    h_threshold_angle_2 = range(0, 30)
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
            if int(theta) in v_threshold_angle and length >= v_threshold_length:
                cv2.line(hoppa, (x1, y1), (x2, y2), (0, 0, 255), 1)
                v_filter.append([x1, y1, x2, y2])
            if (int(theta) in h_threshold_angle_1 or int(theta) in h_threshold_angle_2) and length >= h_threshold_length:
                cv2.line(hoppa, (x1, y1), (x2, y2), (255, 0, 0), 1)
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
            if abs(x1 - x2) < 20 and abs(y1 - y2) < 20:
                indexes.append(j)
    indexes = sorted(set(indexes), reverse=True)
    for i in indexes:
        pts.remove(pts[i])
    filtered_pts = []
    filtered = []
    for pt in pts:
        if pt[0] < 300:
            filtered.append(pt)
    size = len(filtered)
    sorted_filtered = sorted(filtered, key=lambda l: l[1], reverse=False)
    while size > 2:
        sorted_filtered.remove(sorted_filtered[0])
        size = len(sorted_filtered)
    for pt in sorted_filtered:
        filtered_pts.append([pt[0], pt[1]])
    for pt in pts:
        if pt[0] > 1000:
            filtered_pts.append([pt[0], pt[1]])
    filtered_pts = sorted(filtered_pts, key=lambda l: l[1], reverse=False)
    for pt in filtered_pts:
        cv2.circle(hoppa, (pt[0], pt[1]), 10, (0, 255, 0), 2)
    if DEBUG:
        hoppa = cv2.resize(hoppa, (500, 500), interpolation=cv2.INTER_AREA)
        cv2.imshow('sds', hoppa)
        cv2.waitKey(0)
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
            tmp_selected_cont = []
            _range = 3
            i = -1
            while i < _range:
                i += 1
                x, y, w, h = cv2.boundingRect(cntss_sorted_circles[i])
                for X in tmp_selected_cont:
                    if abs(x - X[0]) <= 10:
                        x = 0
                        break
                if x < 100:
                    _range += 1
                    continue
                tmp_selected_cont.append([x, y, h, w])
            tmp_selected_cont = sorted(tmp_selected_cont, key=lambda l: l[0], reverse=False)
            ans = []
            prevX = 0
            prevY = 0  # default y value is about 13
            prevW = 0
            prevH = 0
            for x, y, h, w in tmp_selected_cont:
                if (abs(prevY - y) > 5 or h*w < 600 or h*w > 1500) and prevY != 0:
                    y = prevY
                    x = prevX + 45  # difference between Xs is about 45
                    w = prevW
                    h = prevH
                elif abs(prevY - y) > 5 and prevY == 0 and prevX == 0:
                    i = 1
                    while h*w < 600 and i < 4:
                        x = tmp_selected_cont[i][0] - i * 45  # difference between Xs is about 45
                        y = tmp_selected_cont[i][1]
                        w = 25
                        h = 37
                        i += 1
                cv2.rectangle(questions[k][j], (x, y), (x + w, y + h), (0, 255, 0), 2)
                rect = questions[k][j]
                rect = rect[y:y + h, x:x + w]
                mean = rect.mean()
                prevX = x
                prevY = y
                prevW = w
                prevH = h
                # if h*w < 800:
                #     mean = 300
                ans.append([x, mean])
            if DEBUG:
                cv2.drawContours(questions[k][j], cntss_sorted_circles, -1, (0, 0, 255), 1)
                cv2.imshow('fsad', questions[k][j])
                cv2.waitKey(0)
            sorted_ans = sorted(ans, key=lambda l: l[0], reverse=False)
            while len(sorted_ans) > 4:
                sorted_ans = sorted_ans[1:]
                # print(len(sorted_ans))
            final_answer = []
            tmp_ans = []
            for i in range(0, len(ans)):
                if sorted_ans[i][1] < 190:
                    final_answer.append(answers[i])
                    tmp_ans.append(sorted_ans[i])
            x = -1
            if len(final_answer) == 1:
                if final_answer[0] == model_answers[k + 1 + question_number_offset + j]:
                    total_grade += 1
                else:
                    faults += 1
            elif len(final_answer) > 1:
                # avg = 0
                # for anss in ans:
                #     avg += anss[1]
                # avg /= 4
                # x = 0
                # for i in range(0, 4):
                #     if ans[i][1] < avg/2:
                #         if answers[i] == model_answers[k + 1 + question_number_offset + j]:
                #             total_grade += 1
                #         x += 1
                if abs(tmp_ans[0][1] - tmp_ans[1][1]) > 20:
                    if tmp_ans[0][1] > tmp_ans[1][1]:
                        final_answer.remove(final_answer[0])
                    else:
                        final_answer.remove(final_answer[1])
                    if final_answer[0] == model_answers[k + 1 + question_number_offset + j]:
                        total_grade += 1
                    else:
                        faults += 1
                else:
                    faults += 1
                    x = 0
            elif len(final_answer) < 1:
                x = 0
            if x == 0:
                print("el asshole ele masa7 el x: check el so2al da:", (k + 1 + question_number_offset + j))


            # print("Question", (k + 1 + question_number_offset + j), ":", final_answer)
        question_number_offset += 14
    print("File:", filename)
    print("Total Grade:", total_grade, ", No of Faults:", faults)

    write_list.append(filename)
    write_list.append(str(total_grade))
    toWrite.append(write_list)
    total_grade = 0
    faults = 0
    write_list = []

print(fileCount)
myfile = open('output submission.csv', 'w', newline='')
wr = csv.writer(myfile)
wr.writerow(['FileName', 'Mark'])
for item in toWrite:
    wr.writerow(item)
myfile.close()
