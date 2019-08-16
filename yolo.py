
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": '/logs/trained_weights_final.h5',
        "anchors_path": '/data_set/annotation/tiny_yolo_clusters.txt',
        "classes_path": '/data_set/annotation/classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name'" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.gpu_num = 1
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # 准备已经训练好的模型
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must b a .h5 file.'

        # 加载已经训练好的模型，如果加载失败，则创建模型，并加载训练好的weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print ('{} model, anchors, and classes loaded.'.format(model_path))

        # 为描绘boxes的边框，准备好颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # 输入值的占位
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        # 调用yolo_eval，计算出boxes，scores，classes
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    # 对图片进行检测
    def detect_image(self, image):
        start = timer()

        # 对输入图片进行尺寸检测，宽高必须都是32的倍数
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            # 将检测图片进行尺寸缩放，并居中放置，确保输入神经网络的训练图片size的统一
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            # 如果没有规定模型的图片输入尺寸，则使用图片原始宽高，将其变换为最接近的32的倍数，作为输入尺寸
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print (image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)      # 最前方添加维度，用来匹配批次数

        # 将图片放入神经网络，得到输出值
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={
                                                               self.yolo_model.input: image_data,
                                                               self.input_image_shape: [image.size[1], image.size[0]],
                                                               K.learning_phase(): 0        # 0 -> test模式；1 -> training模式
                                                           })

        print ('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # 定义字体
        font = ImageFont.truetype(font='/Users/weilaixsy/python/yolo_v3/font/msyh.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 定义边框厚度
        thickness = (image.size[0] + image.size[1]) // 600
        # print (out_classes)
        print (out_boxes)
        # print (len(out_boxes))
        print (out_scores)
        print (list(enumerate(out_classes)))
        for i, c in reversed(list(enumerate(out_classes))):
            print (i, c)
            predicted_class = self.class_names[c]
            print (predicted_class)
            box = out_boxes[i]
            score = out_scores[i]
            print (score)

            if box.shape == (0, 4):
                continue
            print (box[0])

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            # label框的显示不能超出image图片的范围
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 画box的矩形框
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # 画标签的矩形框
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # 写入标签内容
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print (end - start)
        return image

    def close_session(self):
        self.sess.close()

# 对视频进行检测
# 将视频内容转换成image，然后使用detect_image来进行检测
def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print ("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(iamge)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()




















