
import numpy as np

# 此模块的作用，是从原始训练数据所标记的所有boxes中，采用聚类方法，挑选出cluster_number个，作为boxes的代表，记为clusters
class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    # 计算交并比iou
    # 这里的交并比，指的是所有box的集合，与聚类出的clusters的交并比
    # cluster的数量由cluster_number决定
    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    # 计算交并比的均值
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    # 对所有boxes进行聚类处理，最终选择出k个框，作为clusters，来代表所有的boxes
    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed(5)
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)     # dist为计算欧氏距离的函数

            last_nearest = current_nearest
        return clusters

    # 将得到的clusters数据，按照规范写入TXT中
    def result2txt(self, data):
        f = open("/Users/MrZhang/Desktop/yolo_v3/data_set/annotation/tiny_yolo_clusters.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    # 将训练集的TXT文件中的信息进行解析，取得所有boxes的信息
    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            # print (length)
            for i in range(1, length):
                # print (infos[i])
                if infos[i] == '\n': continue       # 处理换行符，否则会将其作为一个infos进行处理，会报错
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    # 把上面所有函数进行整合，通过训练集的TXT文件，找到clusters，并写入新的TXT文件中
    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k = self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]      # 对result进行排序
        self.result2txt(result)
        print ("K anchors:\n {}".format(result))
        print ("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
    cluster_number = 6
    filename = "/Users/weilaixsy/python/yolo_v3/data_set/annotation/datas.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()























