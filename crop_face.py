import cv2
import os
from utile.Timer import timer
from utile.Processor import processor


# FRAME SPLITTER
def tear_frame(file_name):
    cap = cv2.VideoCapture(file_name)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('train/pic' + str(i) + '.png', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


# FACE CROPPER
def crop_face(file_name, save_file_name="no_name.png", width=28, height=28, show=False):
    face_1 = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    face_2 = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    image = cv2.imread(file_name)

    faces = face_1.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        faces = face_2.detectMultiScale(image, 1.3, 5)
    x, y, w, h = faces[0]
    face = image[y:y + h, x:x + w]
    new = cv2.resize(face, (width, height))
    cv2.imwrite(save_file_name, new)
    if show is True:
        cv2.imshow('image', new)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# MULTI-FILE CROPPER
def crop_all_faces_in_folder(folder_name='train/', save_folder='cropped_train/', width=28, height=28):
    if os.path.isdir(folder_name):
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        @timer()
        @processor(
            {crop_face: [[f"{folder_name}{i}", f"{save_folder}{i}", width, height] for i in os.listdir("train")]})
        def crop_and_put():
            pass

        if __name__ == '__main__':
            crop_and_put()

    else:
        raise Exception('file not found.')


if __name__ == '__main__':
    # tear_frame("blah.mp4") sample video
    pass
