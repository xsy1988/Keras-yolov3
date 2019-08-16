
import numpy as np
import keras.backend as K
# import tensorflow as tf
# import tensorflow.keras as keras
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    # 定义各类数据的路径
    annotation_path = '/Users/weilaixsy/python/yolo_v3/data_set/annotation/datas.txt'
    log_dir = '/Users/weilaixsy/python/yolo_v3/logs3/'
    classes_path = '/Users/weilaixsy/python/yolo_v3/data_set/annotation/classes.txt'
    anchors_path = '/Users/weilaixsy/python/yolo_v3/data_set/annotation/yolo_clusters.txt'
    # 获取class_names,num_classes,anchors
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    print ("Seccess to get class_names and anchors!")

    input_shape = (416, 416)    #416 = 13 × 32

    # 根据条件，创建对应的训练模型
    is_tiny_version = len(anchors) == 6
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='/Users/weilaixsy/python/yolo_v3/weights_data/yolov3-tiny.h5')
    else:
        model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='/Users/weilaixsy/python/yolo_v3/weights_data/yolo.h5')

    # TensorBoard用于可视化处理
    logging = TensorBoard(log_dir=log_dir)
    # ModelCheckpoint用于监测需要的值并保存
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # ReduceLROnPlateau用于在loss不再降低时调整学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # EarlStopping用于在过拟合前提前停止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 读取训练数据并做预处理
    # 对数据切分为训练集和验证集
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 使用已冻结大部分层的参数的模型，进行第一步训练，用来得到一个不错的loss
    # 训练自己的数据时，需要自行调节epochs
    if True:
        model.compile(optimizer=Adam(lr=0.005), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch = max(1, num_train//batch_size),
                            validation_data = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps = max(1, num_val//batch_size),
                            epochs = 50,
                            initial_epoch = 0,
                            callbacks = [logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # 将模型各层的数据解冻，继续进行训练，相当于对参数进行微调
    # 此阶段训练时间，可以根据结果的情况延长
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print ('Unfreeze all of the layers.')
        batch_size = 32
        print ('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch = max(1, num_train//batch_size),
                            validation_data = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps = max(1, num_val//batch_size),
                            epochs = 100,
                            initial_epoch = 50,
                            callbacks = [logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


# 获取class_names
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]   # c.strip() 除去首位的空格
    return class_names


# 获取anchors
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# 创建yolo模型
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='/Users/weilaixsy/python/yolo_v3/weights_data/yolo.h5'):
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print ('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 如果使用预先训练好的参数，则加载对应参数
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print ('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            # 已经训练好的参数，可以根据需要对某一些层的参数继续训练，其他层的参数冻结掉
            # freeze_body = 2时，冻结层数为"len(model_body.layers)-3"，即除了3个输出层都冻结
            # freeze_body = 1时，冻结层数为185，即冻结darknet53的所有参数
            for i in range(num): model_body.layers[i].trainable = False
            print ('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments = {'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


# 创建tiny_yolo模型
def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='/Users/weilaixsy/python/yolo_v3/weights_data/yolov3-tiny.h5'):
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print ('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 如果使用预先训练好的参数，则加载对应参数
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            # 已经训练好的参数，可以根据需要对某一些层的参数继续训练，其他层的参数冻结掉
            # freeze_body = 2时，冻结层数为"len(model_body.layers)-2"，即除了2个输出层都冻结
            # freeze_body = 1时，冻结层数为20，即冻结darknet的所有参数
            for i in range(num): model_body.layers[i].trainable = False
            print ('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


# 定义数据发生器函数，用来读取原始数据，并给出可用于神经网络的训练数据
# 生产出来的数据：[image_data, y_true]
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)     # 打乱数据
            # 取数据前，对图片做随机的变换、缩放、调整色值等预处理
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # 对box进行预处理
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# 对数据发生器函数做一个包装，排除特殊情况，验证输入参数的正确性
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()























