import cv2 as cv
import os

face_cascade = cv.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_alt.xml")
eyes_cascade = cv.CascadeClassifier("resources/haarcascades/haarcascade_eye.xml")


def detectAndDisplay(img):
    detected = 0
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.equalizeHist(img_gray)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 3)
    # print(faces)
    if len(faces) > 0:
        detected += 1
        # print("face detected")

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        img = cv.ellipse(img, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = img_gray[y:y + h, x:x + w]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.1, 3)
        # print(eyes)
        if len(eyes) > 0:
            detected += 1
            # print("eyes detected")
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            img = cv.circle(img, eye_center, radius, (255, 0, 0), 4)

    # cv.imshow('Capture - Face detection', img)
    return detected


main_dir = "output"
arr = os.listdir(main_dir)
# print(arr)
result = []
for face in arr:
    if not face.startswith('.'):
        face_path = os.path.join(main_dir, face)
        # print(face_path)
        img = cv.imread(face_path)
        res = detectAndDisplay(img)
        result.append(res)

# print(result)
print("total face: ", len(result))
detected_face = len([i for i in result if i > 0])
print("total detected face: ", detected_face)
rec_rate = (detected_face / len(result)) * 100
# cv.waitKey(0)
print("Recognition rate = ", rec_rate)
