#from roboflow import Roboflow
#
#rf = Roboflow(api_key="3rmMtOAwDaAE0wm9EqQ3")
#project = rf.workspace("detectionplayers").project("ball-detect-w4l13")
#version = project.version(3)
#dataset = version.download("yolov11")
                

from ultralytics import YOLO

# Load a model
model = YOLO("C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\weights.onnx")

# Customize validation settings
validation_results = model.val(data="C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\data\\dataset2\\data.yaml", imgsz=1792)                