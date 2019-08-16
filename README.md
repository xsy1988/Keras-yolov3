# Keras-yolo3 中文学习笔记
## 简述
>在学习qqwweee的Keras-yolov3时，根据自己的理解加了一些中文注释，帮助自己理解代码具体实现了什么。     
>qqwweee的原文地址：https://github.com/qqwweee/keras-yolo3     
>>不管是DeepLearning还是敲代码，都刚刚才开始学习，里面应该会有不少错误。     
>但程序是跑起来了，而且使用自己的数据，也获得了不错的训练结果。     

## 学习心得
  >1.不能把yolo v3当做一个黑盒，仅知道如何换上自己的训练数据。最好是能每一行代码敲下来，尽量弄明白各功能是怎么实现的。虽然我仍然还有很多没整明白。  
  >2.注意qqwweee的原文中有一些bug，需要在训练时修正。   
  >3.舍得花时间。   

## 我的学习步骤
### 1.理解代码
  >先弄明白核心代码的内容，建议顺序：yolo3/utils.py --> yolo3/model.py --> train.py --> yolo.py  
  >上面四个文件内的代码学习完后，就对yolo v3的实现机制有了基本的理解  
  >在学习 yolo3/model.py 前，建议先阅读这篇文章：https://blog.csdn.net/leviopku/article/details/82660381 ，非常有帮助  
### 2.制作训练集
  >工具：labelImg  
  >地址：https://github.com/tzutalin/labelImg  
  >自己采集好训练用的图片后，使用labelImg进行标注，得到每张图片的xml文件  
### 3.生成训练用的文件
  得到的xml文件中，包含了训练需要的各类信息，我们需要将其转化成darknet可以使用的数据格式  
  训练时，我们需要三个txt格式的文件：  
    classes.txt ==> 用来存储数据集的类别  
                    在使用labelImg标注图片后，会直接得到这个文件  
    datas.txt   ==> 用来存储所有的数据内容，txt内每一行代表一张图片的标注信息  
                    每一行的数据格式：image_file_path box1 box2 ... boxN  
                    每个box的数据格式：x_min,y_min,x_max,y_max,class_id  
                    例：/train_img/162.png 55,169,92,231,0 127,242,176,301,0 361,157,396,208,0  
                    使用 ”xml_to_txt.py“ ，将标注获得的xml文件转化为datas.txt  
    clusters.txt ==> 在datas.txt中，记录了很多的box数据，它们拥有不同的宽高值
                     我们需要对这些数据先做一个聚类处理，使用”kmeans.py“，挑选出k个宽高值，作为所有框的代表
                     即选出先验框  
### 4.获得已经预先训练好的权重
  从yolo_v3的官网，下载已经训练过的weights：https://pjreddie.com/darknet/yolo/  
  官网上有详细的教程，按照指导下载即可  
  需要注意的是，官网上可以下载的weights有很多种，对应不同的模型，根据自己的需求下载，一定不要弄混了  
  下载后的文件，是类似”yolov3.weights“这种文件，需要将其转化为.h5文件后，才可以在训练中加载读取  
  使用”convert.py“文件进行转化操作，即可获得”yolo.h5“这类文件，用于训练  
### 5.训练
  以上都准备好之后，就可以在"train.py"中，进行训练了  
  我在训练的时候，先使用小批量的数据，跑了20个epoch，看看收敛情况，并对learning_rate做了调整  
  确保一切都顺利后，扩大数据量，跑50至100个epoch  
  训练yolo_tiny可能需要3至4小时，训练yolo可能需要一整天的时间，请一定安排好自己的时间  
### 6.测试
  训练完成后，会获得训练后的weights.h5文件   
  验证时加载训练获得的权重，对test用的image进行测试  
  测试使用 ”detect_img.py“   
 
## 学习结果
  我并没有使用yolov3来检测常规的动物、人物、物体，而是想训练后检测手写字。  
  制作的训练集包含680张图片，每张图内5~18个目标，分类就是手写的字母：A、B、C、D  
### loss
  训练时的loss一直很高，最低也有20+，原本一直认为是有问题，但在实际测试时，发现效果不错  
  为什么loss这么高，但效果却很好，原因暂时不懂  
  
### 实测图
  
  
  
  
  
  
  
  
