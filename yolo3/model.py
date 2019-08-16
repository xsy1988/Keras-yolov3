from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from yolo3.utils import compose

@wraps(Conv2D)  # 装饰器，用于装饰Conv2D，以后调用Conv2D时，会实际调用装饰后的函数，即DarknetConv2D
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}                                  # 正则化采用l2
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'    # 根据strides的值确定padding采用valid还是same
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

# 定义DBL模块，每个模块带有BN和LeakyReLU的模块，在卷积后，跟着一个批量归一化操作，接着是LeakyReLU层
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

# 定义一个resn模块，num_blocks即这个resn中包含多少个res_unit模块
# 每个res_unit模块，2个DBL模块加上输入值
# num_filters为此模块的filter数量，即输出的channel值
# x是输入值
# 每个resn模块包含的操作：先zero_padding，后面接一个DBL模块，再接num_blocks个res_unit模块
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0), (1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x, y])
    return x

# 搭建darknet的主体，其中包含52个卷积层
# 结构为：DBL+res1+res2+res8+res8+res4
def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

# 在darknet主体之后，接着是一个包含5个DBL的模块，然后再接一个包含1个DBL和一个conv的模块，当做输出层
def make_last_layers(x, num_filters, out_filters):
     x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1))
            )(x)
     y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D(out_filters, (1, 1))
            )(x)
     return x, y

# 搭建yolo主体
# 输出分为三类：13×13，26×26，52×52，对应不同的精确度，越高精确度能识别更小的物体，但速度会降低
def yolo_body(inputs, num_anchors, num_classes):

    # 先搭建darknet
    darknet = Model(inputs, darknet_body(inputs))

    # 13×13的输出
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    # 26×26的输出
    # 对13×13中得到的x进行DBL及上采样操作
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2)
            )(x)
    # 拼接张量，x与darknet第152层输出的张量进行拼接
    # darknet第152层张量，即DBL+res1+res2+res8+res8后的输出，后面的res4不参与此计算
    # darknet层数量计算：
        # DBL -> 3层；
        # res_unit -> 3+3+1=7层（add操作算1层）
        # res1 -> 1+3+7 = 11层
        # res2 -> 1+3+7*2 = 18层
        # res8 -> 1+3+7*8 = 60层
        # res4 -> 1+3+7*4 = 32层
    # 故DBL+res1+res2+res8+res8层数为3+11+18+60+60 = 152层
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    # 52×52的输出
    # 对26×26中得到的x进行DBL及上采样操作
    # 取darknet中DBL+res1+res2+res8的输出进行拼接，层数为3+11+18+60 = 92层
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2)
            )(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])

# 搭建简化版的tiny_yolo主体
# 模块1：DBL+MaxP+DBL+MaxP+DBL+MaxP+DBL+MaxP+DBL
# 模块2：MaxP+DBL+MaxP+DBL+DBL
# 输出1：在模块2后接：DBL+Conv
# 输出2：在模块2后接：DBL+UpSampling，然后与模块1拼接，最后接：DBL+Conv
def tiny_yolo_body(inputs, num_anchors, num_classes):
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3))
            )(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1))
            )(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1))
            )(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2)
            )(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1))
            )([x2, x1])

    return Model(inputs, [y1, y2])

# 根据yolo主体的输出，计算出各网格区域内，包含各类锚框的可能性，box的位置及size，以及所标识内容所属class的概率值
# 这里计算出的是所有box，之后再进行筛选
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)      #锚框个数

    # 将锚框数据转化为tensor，其维度为[1,1,1,num_anchors,2]，即[batch, height, width, num_anchors, box_params]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 网格数据，计算损失时使用
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # 对feats做reshape处理，[-1,height,width,num_anchors,num_classes+5]
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes+5])

    # box的属性：point（xy）、宽高（wh）、置信度、所属类
    # 使用reshape后feats最后一维的相关数据（num_classes+5），来计算box的属性
    # 第0、1数据对应point(xy)
    # 第2、3数据对应宽高(wh)
    # 第4数据对应置信度confidence
    # 第5及之后的数据，对应各class的归属概率值
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 如果计算损失，传回grid, feats, box_xy, box_wh
    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    # 如果用作预测，传回box_xy, box_wh, box_confidence, box_class_probs
    return box_xy, box_wh, box_confidence, box_class_probs

# 对box进行位置、大小的纠偏
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))   # 此处使用的是list的min方法，返回list中各元素的最小值
    # 位移偏置值offset
    offset = (input_shape - new_shape) / 2. / input_shape
    # 缩放比例scale
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # box_yx / box_hw，是box的中心点y/x的值，及宽高值，需要转化为左上角及右下角两个顶点的值
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
                box_mins[..., 0:1],     # y_min
                box_mins[..., 1:2],     # x_min
                box_maxes[..., 0:1],    # y_max
                box_maxes[..., 1:2]     # x_max
                ])

    # 将计算出的boxes两个顶点的四个值，放大到image的大小中。
    # ？？？由于box_yx 变换为y_min,x_min,y_max,x_max,维度的取值由2拓展到4，所以计算时image也需要做对应维度取值增加？？？
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# 计算出每个box的得分
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

# yolo_outputs : yolo_body -> [y1,y2,y3]; tiny_yolo_body -> [y1,y2]
def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    num_layers = len(yolo_outputs)
    # num_layers=3 -> yolo
    # num_layers=2 -> tiny_yolo
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers==3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # 对每个尺度的layer，计算出其预测出的box，及box的scores
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 用拼接展平
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # 创建掩码
    mask = box_scores >= score_threshold
    # max_boxes转化为Tensor
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    # 对每一个class，筛选出其对应的box
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 使用官方的内置函数，筛选出符合条件的box所在的位置
        # nms:非最大值抑制
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        # 使用官方的内置函数，获得筛选出来的box
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        # 记录当前box所属的类
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

# 在训练前，对训练数据中的box进行预处理，使true_boxes数据转换为y_true数据
# true_boxes: array, shape=(m,T,5)
# input_shape: array-like, hw, *32
# anchors: array, shape=(N, 2), wh
# y_true: array, shape=shape(yolo_outputs)
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all()
    num_layers = len(anchors) // 3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]      # [[13,13],[26,26],[52,52]]
    # y_true -> [(m, 13, 13, 3, 5+num_classes),(m, 26, 26, 3, 5+num_classes),(m, 52, 52, 3, 5+num_classes)]
    y_true = [np.zeros((m,grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes), dtype='float32') for l in range(num_layers)]

    # anchors增加1维，(9,2) -> (1,9,2)
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue

        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype('int32')
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1
                    y_true[l][b,j,i,k,5+c] = 1

    return y_true

# 计算交并比iou
# b1: tensor, shape=(i1,...,iN,4), xywh
# b2: tensor, shape=(j,4), xywh
# iou: tensor, shape=(i1,...,iN,j)
def box_iou(b1, b2):
    b1 = K.expand_dims(b1, -2)      # shape=(i1,...,iN,1,4)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = K.expand_dims(b2, 0)       # shape=(j,1,4)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算b1、b2相交的面积，进而计算出交并比
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

# 计算yolo的loss
# loss: tensor, shape=(1,)
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    num_layers = len(anchors) // 3
    yolo_outputs = args[: num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]     # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss















