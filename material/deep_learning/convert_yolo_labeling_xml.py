%cd /content/keras-YOLOv3-model-set

import os
import glob

CLASS_NAMES = [ "cat", "dog" ] 
# TAGET_FOLDER_NAME = "labels"
TAGET_FOLDER_NAME = "dogs_cats_yolo_labeled"


label_file_names = []
for file_name in glob.glob(TAGET_FOLDER_NAME+'/*.xml'):
  label_file_names.append(file_name)

print(len(label_file_names))
print(label_file_names)

import xml.etree.ElementTree as ET

all_record = []

for file_name in label_file_names:

  with open(file_name) as in_file:

    tree = ET.parse(in_file)
    root = tree.getroot()

    a_record = [ file_name.replace("xml", "jpg") ]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASS_NAMES or int(difficult)==1:
            continue
        cls_id = str(CLASS_NAMES.index(cls))
        xmlbox = obj.find('bndbox')
        xmin = xmlbox.find('xmin').text
        ymin = xmlbox.find('ymin').text
        xmax = xmlbox.find('xmax').text
        ymax = xmlbox.find('ymax').text

        a_record.append(",".join([xmin,ymin,xmax,ymax,cls_id]))

    all_record.append(a_record)


with open("labels.txt", "w") as f:
  for a_record in all_record:
    print(a_record)
    f.write(" ".join([str(i) for i in a_record]))
    f.write("\n")

print("labels.txt created.")
