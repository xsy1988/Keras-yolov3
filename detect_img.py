import sys
from yolo import YOLO
from PIL import Image

#
def detect_img(yolo, image):
    r_img = yolo.detect_image(image)
    r_img.show()
    yolo.close_session()

model_path = "/logs/trained_weights_final.h5"
anchors_path = "/data_set/annotation/yolo_clusters.txt"
classes_path = "/data_set/annotation/classes.txt"
model_image_size = (416, 416)
gpu_num = 1

if __name__ == '__main__':

    yolo = YOLO(model_path=model_path, anchors_path=anchors_path, classes_path=classes_path, model_image_size=model_image_size)
    image_path = "/data_set/test_img/test1.jpg"
    image = Image.open(image_path)
    detect_img(yolo, image)
