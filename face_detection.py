import cv2


face_id = cv2.CascadeClassifier('haar cascade/haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)

#I'm using path of the video because the camera of my laptop is damaged
# To use your laptop camera, comment out the above line of code and uncomment the line below this comment
# vid = cv2.VideoCapture(0)

while True:
    ret, frames = vid.read()
    #parameters of muktiscale are input frames,scale factor(greater than 1),minimum neighbours)
    face_det = face_id.detectMultiScale(frames,1.2, 9)
    for (x, y, w, h) in face_det:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('Face Recognition', frames)

    # The video can be stopped when we press 'w' on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

vid.release()
cv2.destroyAllWindows()
   