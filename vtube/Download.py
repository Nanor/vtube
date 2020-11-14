import os
import requests

def download_model(name):
    if name == 'posenet':
        url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
    else:
        url = "https://github.com/google/mediapipe/raw/master/mediapipe/modules/{}/{}.tflite".format(name, name)

    if not os.path.isdir("data"):
        os.mkdir("data")

    if name not in os.listdir("data"):
        r = requests.get(url)
        open("data/{}.tflite".format(name), "wb").write(r.content)
