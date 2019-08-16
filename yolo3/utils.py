# import functools

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 给定一批函数，按照规则将其组合起来
def compose(*funcs):

    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
        # 函数f,g，参数*a,**kw，funcs为需要组合的函数集合
        # *a 是一个元组，参数数量未知，如(1，2，3，4，5)
        # **kw 是一个词典，参数数量未知，如(A='a', B='b', C='c')
    else:
        raise ValueError('Composition of empty sequence not supported')

# 给定一个size，将原来的image进行适当缩放，置中放置于对应size灰色图相框中
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

# [a,b]之间随机均匀取值
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# 对图片进行预处理
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # 如果不是随机变换图片，则按input_shape来对应缩放并置中放置图片
    if not random:
        scale = min(w/iw, h/ih)
        hw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_iamge.paste(iamge, (dx, dy))
            image_data = np.array(new_image)/255.

        # 对相应的box也进行缩放和位置偏移
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            # box format: [x_min, y_min, x_max, y_max, class_id]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx  # [0, 2] --> [x_min, x_max]
            box[:, [1,3]] = box[:, [1,3]]*scale + dy  # [1, 3] --> [y_min, y_max]
            box_data[:len(box)] = box

        return image_data, box_data

    # 对图片做随机缩放，并以随机位置放置于图片框中
    new_ar = w/h * rand(1 - jitter, 1+ jitter) / rand(1 - jitter, 1 + jitter)  # jitter=0.3，则中心点在1，左右各宽0.3, [0.7, 1.3]之间随机均匀取值
    scale = rand(.25, 2)  # [0.25, 2]之间随机均匀取值
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw*new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 随机左右翻转图片
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 对图片做随机色值调整
    # HSV色值体系中： H -> 色调，S -> 饱和度，V -> 亮度
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x)

    # 对相应的box也做缩放和位置偏移
    box_data = np.zeros((max_boxes, 5))
    if len(box)>0:
        np.random.shuffle(box)
        # 缩放及位置偏移
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        # 如果图片左右变换了，box也需要对应变换
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        # 对box的x、y值做最大最小值约束
        box[:, 0:2][box[:, 0:2]<0] = 0  #x_min, y_min 最小值为0
        box[:, 2][box[:,2]>w] = w       #x_max 最大值为w
        box[:, 3][box[:,3]>h] = h       #y_max 最大值为h
        box_w = box[:, 2] - box[:, 0]   #box的宽
        box_h = box[:, 3] - box[:, 1]   #box的高
        box = box[np.logical_and(box_w>1, box_h>1)] # 只有宽高均大于1的box被留下，其他被舍弃
        if len(box)>max_boxes: box = box[: max_boxes]
        box_data[: len(box)] = box

    return image_data, box_data

