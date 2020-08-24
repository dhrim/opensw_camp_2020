import os
import glob

CLASS_NAMES = [ "cat", "dog" ] 
TAGET_FOLDER_NAME = "labels"

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

    a_record = [ file_name.replace("xml", "jpg" ]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASS_NAMES or int(difficult)==1:
            continue
        cls_id = CLASS_NAMES .index(cls)
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)

        a_record.append(xmin)
        a_record.append(ymin)
        a_record.append(xmax)
        a_record.append(ymax)
        a_record.append(cls_id)

    all_record.append(a_record)


with open("labels.txt", "w") as f:
  for a_record in all_record:
    print(a_record)
    f.write(" ".join([str(i) for i in a_record]))
    f.write("\n")