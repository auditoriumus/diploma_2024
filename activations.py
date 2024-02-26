import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tf_explain.core.grad_cam import GradCAM
import PIL
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

from argparse import ArgumentParser

import glob
import os

# Используется модель VGG16.
model = VGG16(weights='imagenet', include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000)

# Проверка журналированием print(model.summary())
last_conv_layer_name = "block5_conv3"

# Включаем слои между последним сверточным слоем и слоем прогнозирования
# Название слоев можно найти через журналирование print(model.summary())
classifier_layer_names = ["block5_pool", "flatten", "fc1", "fc2", "predictions"]

# Функция вызывается из 'make_gradcam_heat map'
# Берет image_path из 'get_command_line_arguments' преобразует в массив
def get_img_array(img_path, size):
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # `array` is a float32 Numpy array
    array = tensorflow.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

#'make_gradcam_heatmap' является основной функцией и возвращает тепловую карту, наложенную на входное изображение

# Входными данными являются путь к изображению, указанный в командной строке, последний сверточный слой и
# названия слоев классификатора, которые зависят от модели, и путь вывода для тепловой карты
def make_gradcam_heatmap(
    img_path, model, last_conv_layer_name, classifier_layer_names, output_path
):
    # предварительно обрабатывает массив, возвращаемый из 'get_img_array'
    img_array = preprocess_input(get_img_array(img_path, size= (224, 224)))

    # Создаем модель, которая сопоставляет входное изображение с активациями последнего слоя conv
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tensorflow.keras.Model(model.inputs, last_conv_layer.output)

    # Создаем модель, которая сопоставляет активации последнего диалога слоя с окончательными прогнозами класса
    classifier_input = tensorflow.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tensorflow.keras.Model(classifier_input, x)

    # Вычисляем градиент верхнего прогнозируемого класса для
    # входного изображения относительно активаций последнего слоя conv
    with tensorflow.GradientTape() as tape:

        # Вычисление активации последнего слоя conv и его просмотр
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)

        # Вычисление предсказания класса объекта
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tensorflow.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Градиент верхнего прогнозируемого класса по отношению к выходной карте объектов последнего слоя conv
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Вектор, где каждая запись представляет собой среднюю интенсивность градиента по определенному каналу карты объектов
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

    # Умножаем каждый канал в массиве карт объектов на вес "важности канала" по отношению к высшему прогнозируемому классу
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Среднее значение по каналам результирующей карты объектов - это и есть тепловая карта активации класса
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Для визуализации нормализуем тепловую карту в диапазоне от 0 до 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Загрузка оригинального изображения
    img = tensorflow.keras.preprocessing.image.load_img(img_path)
    img = tensorflow.keras.preprocessing.image.img_to_array(img)

    # Масштаб тепловой карты до диапазона 0-255
    heatmap = np.uint8(255 * heatmap)

    # Раскрашиваем изображение
    jet = cm.get_cmap("jet")

    # Использование значения RGB для цветовой карты
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Создание изображения с цветной тепловой картой RGB
    jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Наложение тепловой карты на исходное изображение
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

    # Сохранение выходного изображения
    superimposed_img.save(output_path)

# Запускает основной код для всего videoframs_path
def process_video(videoframes_path, output_prefix):
    counter = 0
    # Определяет выходную директорию
    output_dir = output_prefix + "_output"

    # Создает выходную директорию, если еще не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for input_path in sorted(glob.glob(videoframes_path + "/*.jpg")):
        counter += 1

        output_path = output_dir + "/result-" + str(counter).zfill(4) + '.jpg'

        # Запускает основную функцию с указанными image_path, output_prefix и слоями
        make_gradcam_heatmap(input_path, model, last_conv_layer_name, classifier_layer_names, output_path)

# Функция для ввода данных через командную строку
def get_command_line_arguments():
    parser = ArgumentParser()
    # Указывается либо изображение, либо видео
    parser.add_argument("--process", choices=["image", "video"], required=True,
                        dest="process_type", help="Process a single image or video")
    parser.add_argument("--path", required=True, dest="path",
                        help="Path of image or directory containing video frames")
    return parser.parse_args()


args = get_command_line_arguments()

# Если process указан как 'image', определяет image_path и output_prefix в соответствии с аргументом командной строки
if args.process_type == "image":
    # Пусть к целевому изображению
    image_path = args.path
    output_prefix = os.path.splitext(os.path.basename(image_path))[0]
    # Запускает основную функцию с указанным image_path и output_prefix из командной строки
    make_gradcam_heatmap(image_path, model, last_conv_layer_name, classifier_layer_names, output_prefix + "_output.jpg")

    # Построение наложенного изображения
    img = mpimg.imread(output_prefix + "_output.jpg")
    plt.imshow(img)
    plt.show()

# Если process указан как 'video', определяет videoframes_path и output_prefix в соответствии с аргументом командной строки
elif args.process_type == "video":
    # Определяем каталог с кадрами, разделенными ffmpeg
    videoframes_path = args.path
    # Используется для конечной директории
    output_prefix = os.path.dirname(videoframes_path)
    # Запускает функцию 'process_video' с входными данными командной строки
    heatmaps = process_video(videoframes_path, output_prefix)
