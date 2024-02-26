import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import os
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


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

out=cv2.VideoWriter('./output.avi',
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20, (frame_width, frame_height))

def initialize_deepsort():
    # загрузка конфигурации глубокой сортировки из YAML-файла
    cfg_deep = get_config()
    cfg_deep.merge_from_file("/content/deep_sort_pytorch/configs/deep_sort.yaml")

    # инициализация трекера на основе глубокой сортировки
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # параметр min_confidence устанавливает минимальную достоверность отслеживания,
                        # необходимую для того, чтобы обнаружение объекта учитывалось в процессе отслеживания
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        # nms_max_overlap указывает максимально допустимое перекрытие между
                        # ограничивающими рамками во время немаксимального подавления (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        # параметр max_iou_distance определяет максимальное расстояние
                        # пересечения через соединение (IoU) между обнаружениями объектов
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # если идентификатор отслеживания объекта потерян, Max_age определяет,
                        # сколько кадров трекер должен подождать, прежде чем присвоить новый идентификатор
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        # nn_budget устанавливает так называемы бюджет для поиска по ближайшему соседу
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deepsort

deepsort = initialize_deepsort()

names = cococlassNames()
colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Function that adds fixed color depending on the class
    """
    if label == 0:  # person  #BGR
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1,  x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # создание самой рамки вокруг обнаруженного объекта
        cv2.rectangle(img, (x1, y1), (x2, y2), color= compute_color_for_labels(cat),thickness=2, lineType=cv2.LINE_AA)
        label = str(id) + ":" + classNames[cat]
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        # создание прямоугольной метки над объектом с указанием уровня достоверности
        t_size=cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)[0]
        c2=x1+t_size[0], y1-t_size[1]-3
        cv2.rectangle(frame, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img

classNames = cococlassNames()
output = cv2.VideoWriter('/content/drive/MyDrive/video_test/output_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

i = 0
while True:
    if i % 5 == 0:
        output.write(frame)
        continue
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    ret, frame = cap.read()
    if ret:
      result = list(model.predict(frame, conf=0.5))[0]
      bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
      confidences = result.prediction.confidence
      labels = result.prediction.labels.tolist()
      for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
          bbox = np.array(bbox_xyxy)
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((confidence*100))/100
          cx, cy = int((x1+x2)/2), int((y1+y2)/2)
          bbox_width = abs(x1-x2)
          bbox_height = abs(y1-y2)
          xcycwh = [cx, cy, bbox_width, bbox_height]
          xywh_bboxs.append(xcycwh)
          confs.append(conf)
          oids.append(int(cls))
      xywhs = torch.tensor(xywh_bboxs)
      confss= torch.tensor(confs)
      outputs = deepsort.update(xywhs, confss, oids, frame)
      if len(outputs)>0:
          bbox_xyxy = outputs[:,:4]
          identities = outputs[:, -2]
          object_id = outputs[:, -1]
          draw_boxes(frame, bbox_xyxy, identities, object_id)
      output.write(frame)
    else:
        break
    i = i+1

output.release()
cap.release()

def display_compressed_video(input_path):
    # Сжимаем видео
    compressed_path = "./result_compressed.mp4"
    os.system(f"ffmpeg -i {input_path} -vcodec libx264 {compressed_path}")


input_video_path = './output_2.avi'
display_compressed_video(input_video_path)
