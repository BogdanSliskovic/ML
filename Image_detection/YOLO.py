import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras import regularizers

import kagglehub

#path = kagglehub.dataset_download("andrewmvd/car-plate-detection")

#RZS
# path = r'C:\Users\bogdan.sliskovic\.cache\kagglehub\datasets\andrewmvd\car-plate-detection\versions\1'
#MRJB
path = r'C:\\Users\\Jelena\\.cache\\kagglehub\\datasets\\andrewmvd\\car-plate-detection\\versions\\1'
os.chdir(path)
os.listdir()
y_dir = os.path.join(path,os.listdir()[0])
x_dir = os.path.join(path,os.listdir()[1])


for folder in [x_dir, y_dir]:
    print(os.listdir(folder)[:5])


for image in sorted(os.listdir(x_dir))[:5]:
    img_path = os.path.join(x_dir, image)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pronađi .xml fajl za tu sliku
    xml_filename = image.split('.')[0] + ".xml"
    xml_path = os.path.join(y_dir, xml_filename)

    # Parsiraj XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Crtaj bounding box i labelu
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.title(image)
    plt.show()


x = []
for fname in os.listdir(x_dir):
    img_path = os.path.join(x_dir, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x.append(img)

x = np.array(x, dtype=np.float32) / 255.0  

x.shape

from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
model.export(format="onnx")
import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load("yolo11n.onnx")  # load onnx model
output = prepare(onnx_model).run(input)  # run the loaded model
tf2onnx.convert.from_onnx
onnx.backhand

model.train(data = )
model
from onnx_tf.backend import prepare