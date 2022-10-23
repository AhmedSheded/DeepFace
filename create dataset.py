import cv2 as cv
import time
import os

os.chdir('data')

num_imges = 300

labels = ['Omar']

def create_dataset(labels, num_imges):

    face_casecade = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    for name in labels:
        output_folder = os.path.join('dataset', name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        print('Cap images for ', name)
        cap = cv.VideoCapture(0)

        count = 0
        time.sleep(30)

        while count<=num_imges:
            ret, frame = cap.read()
            if ret:
                grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = face_casecade.detectMultiScale(grey, 1.3, 5, minSize=(120, 120))

                for x, y, w, h in faces:
                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_img = cv.resize(frame[y:y+h, x:x+w], (152, 152))
                    face_filename = '%s/%d.jpg'% (output_folder, count)
                    cv.imwrite(face_filename, face_img)
                    count +=1
                cv.imshow('Capturing Faces...', frame)
                if cv.waitKey(30) == 27:
                    break
            else:
                break
        cap.release()
        cv.destroyAllWindows()


create_dataset(labels, num_imges)

