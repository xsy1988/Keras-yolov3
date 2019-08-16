import os
import sys
import xml.etree.ElementTree as ET
import glob



def xml_to_txt(indir, outdir):
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')

    f_w = open(outdir, 'w')

    for i, file in enumerate(annotations):

        # file_save = file.split('.')[0] + '.txt'
        # file_txt = os.path.join(outdir, file_save)
        # f_w = open(file_txt, 'w')

        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        annotation_path = root.find('path').text
        f_w.write(annotation_path + ' ')

        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text

            if name == 'a':
                name = 0
            elif name == 'b':
                name = 1
            elif name == 'c':
                name = 2
            else:
                name = 3

            xmlbox = obj.find('bndbox')
            # print (xmlbox)
            # if isinstance(stockInfo, bs4.element.Tag):
            xmin = xmlbox.find('xmin').text
            xmax = xmlbox.find('xmax').text
            ymin = xmlbox.find('ymin').text
            ymax = xmlbox.find('ymax').text

            f_w.write(xmin + "," + ymin + "," + xmax + "," + ymax + ",")
            f_w.write(str(name))
            f_w.write(" ")
        f_w.write('\n')
    f_w.close()


indir = "/data_set/annotation_xml"
outdir = "/data_set/annotation/datas.txt"


if __name__ == '__main__':
    xml_to_txt(indir, outdir)
