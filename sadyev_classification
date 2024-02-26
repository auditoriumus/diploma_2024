import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np

# описываем функцию классификации
def classify(img_path):
    current_img = image.load_img(img_path, target_size=(224, 224))
    model = tensorflow.keras.applications.resnet50.ResNet50()
    img_array = image.img_to_array(current_img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tensorflow.keras.applications.resnet50.preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    result = tensorflow.keras.applications.resnet50.decode_predictions(prediction, top=3)[0]
    print(result)

# передаем относительный путь до файла
classify("./photo_1.jpg")
