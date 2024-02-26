import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import os


def get_video_info(video_path):
    # загрузка видео
    cap = cv2.VideoCapture(video_path)
    # проверка успешного открытия видео
    if not cap.isOpened():
        raise ValueError("Ошибка: Невозможно открыть видеофайл.")
    # получение размеров кадра ширина и длина
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, frame_width, frame_height


def load_model(model_name):
    # Проверка доступен дли CUDA для работы
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Загрузка модели
    model = models.get(model_name, pretrained_weights="coco").to(device)
    return model

model_name = 'yolo_nas_s'
model = load_model(model_name)

classNames = cococlassNames()


out=cv2.VideoWriter('./output.avi',
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20, (frame_width, frame_height))
count=0
while True:
  ret, frame=cap.read()
  count+=1
  if ret:
    result=list(model.predict(frame, conf=0.5))[0]
    bbox_xyxys=result.prediction.bboxes_xyxy.tolist()
    confidences=result.prediction.confidence
    labels=result.prediction.labels.tolist()
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
      bbox=np.array(bbox_xyxy)
      x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      classname=int(cls)
      class_name=classNames[classname]
      conf=math.ceil((confidence*100))/100
      label=f'{class_name}{conf}'
      print("Frame N", count, "", x1, y1,x2, y2)
      # создание ограничительной рамки возкруг объекта
      cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255, 0.6),thickness=2, lineType=cv2.LINE_AA)
      # создание прямоугольной метки над объектом с указанием уровня достоверности
      t_size=cv2.getTextSize(str(label), 0, fontScale=1/2, thickness=1)[0]
      c2=x1+t_size[0], y1-t_size[1]-3
      cv2.rectangle(frame, (x1, y1), c2, color=(0, 0, 255, 0.6), thickness=-1, lineType=cv2.LINE_AA)
      cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    out.write(frame)
  else:
    break
out.release()
cap.release()

def display_compressed_video(input_path):
    # Сжимаем выходное видео
    compressed_path = "./result_compressed.mp4"
    os.system(f"ffmpeg -i {input_path} -vcodec libx264 {compressed_path}")
