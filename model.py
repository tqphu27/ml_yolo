#model_marketplace.config
# {"token_length": "4018", "accuracy": "70", "precision": "fp16", "sampling_frequency:": "44100", "mono": true, "fps": "74", "resolution": "480", "image_width": "1080", "image_height": "1920", "framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name": "DeepSeek-V3",
#       "value": "deepseek-ai/DeepSeek-V3",
#       "size": 20,
#       "paramasters": "685B",
#       "tflops": 14, 
#       "vram": 20,
#       "nodes": 10
#     },
# {
#       "name": "DeepSeek-V3-bf16",
#       "value": "opensourcerelease/DeepSeek-V3-bf16",
#       "size": 1500,
#       "paramasters": "684B",
#       "tflops": 80, 
#       "vram": 48,
#       "nodes": 10
#     }
#   ], "cuda": "11.4", "task":["text-generation", "text-classification", "text-summarization", "text-ner", "question-answering"]}
import io
import time
from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
from ultralytics import YOLO
import os
# from git import Repo
import zipfile
import subprocess
import requests
# from IPython.display import Image
import yaml
import threading
import shutil

import cv2
import numpy as np
from io import BytesIO
from PIL import Image

import asyncio
import logging
import signal
import logging
from huggingface_hub import create_repo, login, upload_folder

from centrifuge import CentrifugeError, Client, ClientEventHandler, SubscriptionEventHandler

from dashboard import promethus_grafana

import base64
import hmac
import json
import hashlib
from gradio_webrtc import WebRTC
from twilio.rest import Client

from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
from mcp.server.fastmcp import FastMCP
import uuid
CHANNEL_STATUS = {}

mcp = FastMCP("aixblock-mcp")
def decode_base64_to_image(base64_str):
    # Kiểm tra xem chuỗi base64 có chứa header không, nếu có thì loại bỏ
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    # Giải mã base64 thành bytes
    image_data = base64.b64decode(base64_str)
    
    # Đọc ảnh từ bytes
    image = Image.open(BytesIO(image_data))
    
    # Chuyển đổi ảnh từ định dạng PIL thành numpy array (vì YOLO dùng numpy arrays)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def mask_to_polygons(mask, max_width, max_height, simplification=0.001):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    # some versions of cv2 does not support incontiguous arr
    mask = np.ascontiguousarray(mask)
    contours, hierarchy = cv2.findContours(mask.astype(
        "uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = contours[0]
    print(f"Befor approxPolyDP: {len(res)}")
    epsilon = simplification * cv2.arcLength(res, True) #simplification * 
    res = cv2.approxPolyDP(res, epsilon, True)
    print(f"After approxPolyDP: {len(res)}")
    res = res.reshape(-1, 2)

    # # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # # We add 0.5 to turn them into real-value coordinate space. A better solution
    # # would be to first +0.5 and then dilate the returned polygon by 0.5.
    # # res = [list(x + 0.5) for x in res if len(x) >= 6]
    res = list(res+0.25) if len(res) >= 6 else []
    # # res = [[res[i]/max_width*100, res[i+1]/max_height*100]
    # #        for i in range(0, len(res), 2)]
    if res:
        x, y = zip(*res)
        x = [i/max_width*100 for i in x]
        y = [i/max_height*100 for i in y]
        res = list(zip(x, y))

    return res, has_holes


HOST_NAME = "https://dev-us-west-1.aixblock.io"
TYPE_ENV = os.environ.get('TYPE_ENV',"DETECTION")

try:
    account_sid = "AC10d8927d012a31a1f5b1696d2e323a4b"
    auth_token = "7ca8d300f9850ddba5919c2ef4df2648"

    if account_sid and auth_token:
        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        rtc_configuration = {
            "iceServers": token.ice_servers,
            "iceTransportPolicy": "relay",
        }
    else:
        rtc_configuration = None
except:
    rtc_configuration = None

css = """
@import "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css";

.aixblock__title .md p {
    display: block;
    text-align: center;
    font-size: 2em;
    font-weight: 700;
}

.aixblock__tabs .tab-nav {
    justify-content: center;
    gap: 8px;
    padding-bottom: 1rem;
    border-bottom: none !important;
}

.aixblock__tabs .tab-nav > button {
    border-radius: 8px;
    border: 1px solid #DEDEEC;
    height: 32px;
    padding: 8px 10px;
    text-align: center;
    line-height: 1em;
}

.aixblock__tabs .tab-nav > button.selected {
    background-color: #5050FF;
    border-color: #5050FF;
    color: #FFFFFF;
}

.aixblock__tabs .tabitem {
    padding: 0;
    border: none;
}

.aixblock__tabs .tabitem .gap.panel {
    background: none;
    padding: 0;
}

.aixblock__input-image,
.aixblock__output-image {
    border: solid 2px #DEDEEC !important;
}

.aixblock__input-image {
    border-style: dashed !important;
}

footer {
    display: none !important;
}

button.secondary,
button.primary {
    border: none !important;
    background-image: none !important;
    color: white !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}

button.primary {
    background-color: #5050FF !important;
}

button.secondary {
    background-color: #F5A !important;
}

.aixblock__input-buttons {
    justify-content: flex-end;
}

.aixblock__input-buttons > button {
    flex: 0 !important;
}

.aixblock__trial-train {
    text-align: center;
    margin-top: 2rem;
}
"""

js = """
window.addEventListener("DOMContentLoaded", function() {
    function process() {
        let buttonsContainer = document.querySelector('.aixblock__input-image')?.parentElement?.nextElementSibling;
        
        if (!buttonsContainer) {
            setTimeout(function() {
                process();
            }, 100);
            return;
        }
        
        document.querySelectorAll('.aixblock__input-image').forEach(function(ele) {
            ele.parentElement.nextElementSibling.classList.add('aixblock__input-buttons');
        });
    }
    
    process();
});
"""

# def download_checkpoint(weight_zip_path, project_id, checkpoint_id, token):
#     url = f"{HOST_NAME}/api/checkpoint_model_marketplace/download/{checkpoint_id}?project_id={project_id}"
#     payload = {}
#     headers = {
#         'accept': 'application/json',
#         # 'Authorization': 'Token 5d3604c4c57def9a192950ef7b90d7f1e0bb05c1'
#         'Authorization': f'Token {token}'
#     }
#     response = requests.request("GET", url, headers=headers, data=payload) 
#     checkpoint_name = response.headers.get('X-Checkpoint-Name')

#     if response.status_code == 200:
#         with open(weight_zip_path, 'wb') as f:
#             f.write(response.content)
#         return checkpoint_name
    
#     else: 
#         return None

# def download_dataset(data_zip_dir, project_id, dataset_id, token):
#     # data_zip_path = os.path.join(data_zip_dir, "data.zip")
#     url = f"{HOST_NAME}/api/dataset_model_marketplace/download/{dataset_id}?project_id={project_id}"
#     payload = {}
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Token {token}'
#     }

#     response = requests.request("GET", url, headers=headers, data=payload)
#     dataset_name = response.headers.get('X-Dataset-Name')
#     if response.status_code == 200:
#         with open(data_zip_dir, 'wb') as f:
#             f.write(response.content)
#         return dataset_name
#     else:
#         return None

# def upload_checkpoint(checkpoint_model_dir, project_id, token):
#     url = f"{HOST_NAME}/api/checkpoint_model_marketplace/upload/"

#     payload = {
#         "type_checkpoint": "ml_checkpoint",
#         "project_id": f'{project_id}',
#         "is_training": True
#     }
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Token {token}'
#     }

#     checkpoint_name = None

#     # response = requests.request("POST", url, headers=headers, data=payload) 
#     with open(checkpoint_model_dir, 'rb') as file:
#         files = {'file': file}
#         response = requests.post(url, headers=headers, files=files, data=payload)
#         checkpoint_name = response.headers.get('X-Checkpoint-Name')

#     return checkpoint_name

from typing import List, Dict, Any
from shapely.geometry import Polygon

def calculate_iou(polygon1: List[List[float]], polygon2: List[List[float]]) -> float:
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union != 0 else 0

def nms_for_polygons(results: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    # Sắp xếp các phân đoạn theo `avg_score` giảm dần
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    filtered_results = []

    for i, result in enumerate(results):
        keep = True
        for filtered in filtered_results:
            iou = calculate_iou(result["value"]["points"], filtered["value"]["points"])
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            filtered_results.append(result)
    
    return filtered_results

class MyModel(AIxBlockMLBase):

    is_init_model_trial = False
    model_trial_public_url = ""
    model_trial_local_url = ""

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return []

    def fit(self, event, data, **kwargs):
        """

        
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

    def action(self, command, **kwargs):
        print(f"""
                command: {command},
                kwargs: {kwargs}
            """)
        if command.lower() == "train":
            try:
                clone_dir = os.path.join(os.getcwd())

                epochs = kwargs.get("epochs", 20)
                imgsz = kwargs.get("imgsz", 640)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset_id = kwargs.get("dataset")
                push_to_hub = kwargs.get("push_to_hub", True)
                hf_model_id = kwargs.get("hf_model_id", "deepseek-v3-1b")
                channel_log = kwargs.get("channel_log", "training_logs")

                push_to_hub_token = kwargs.get("push_to_hub_token", "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")
                
                log_queue, logging_thread = start_queue(channel_log)
                write_log(log_queue)

                channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
                username = ""
                hf_model_name = ""

                try:
                    headers = {"Authorization": f"Bearer {push_to_hub_token}"}
                    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        username = data.get("name")
                        hf_model_name = f"{username}/{hf_model_id}"
                        print(f"Username: {username}")
                    else:
                        print(f"Error: {response.status_code} - {response.text}")
                        hf_model_name = "Token not correct"
                except Exception as e:
                    hf_model_name = "Token not correct"
                    print(e)

        
                # Đặt trạng thái kênh là "training"
                CHANNEL_STATUS[channel_name] = {
                        "status": "training",
                        "hf_model_id": hf_model_name,
                        "command": command,
                        "created_at": time.time()
                    }
                print(f"🚀 Đã bắt đầu training kênh: {channel_name}")
                
                def func_train_model(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id):
                    print("Giá trị", HOST_NAME, token, project_id)
                    project = connect_project(HOST_NAME, token, project_id)
                    os.makedirs(f'{clone_dir}/data_zip', exist_ok=True)
                    os.makedirs(f'{clone_dir}/models', exist_ok=True)

                    weight_path = os.path.join(clone_dir, f"models")
                    dataset_path = os.path.join(clone_dir, f"datasets/dataset0") 
                    data_train_dir = os.path.join(dataset_path, "dota8.yaml")


                    if checkpoint_version and checkpoint_id:
                        weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                        if not os.path.exists(weight_path):
                            weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                            checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                            if checkpoint_name:
                                # weight_path = os.path.join(clone_dir, f"models/{checkpoint_name}")
                                # if not os.path.exists(weight_path):
                                with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(weight_path)

                    if dataset_version and dataset_id:
                        dataset_path = os.path.join(clone_dir, f"datasets/{dataset_version}")
                        if not os.path.exists(dataset_path):
                            # data_zip_dir = os.path.join(clone_dir, "data_zip/data.zip")
                            # dataset_name = download_dataset(data_zip_dir, project_id, dataset_id, token)
                            data_path = os.path.join(clone_dir, "data_zip")
                            os.makedirs(data_path, exist_ok=True)
                            dataset_name = download_dataset(project, dataset_id, data_path)
                            if dataset_name: 
                                data_zip_dir = os.path.join(data_path, dataset_name)

                                # Giải nén file đầu tiên
                                with zipfile.ZipFile(data_zip_dir, 'r') as zip_ref:
                                    zip_ref.extractall(dataset_path)

                                # Kiểm tra nếu trong dataset_path chỉ có 1 file zip => giải nén tiếp
                                extracted_files = os.listdir(dataset_path)
                                zip_files = [f for f in extracted_files if f.endswith('.zip')]

                                if len(zip_files) == 1:
                                    inner_zip_path = os.path.join(dataset_path, zip_files[0])
                                    print(f"🔁 Found inner zip file: {inner_zip_path}, extracting...")
                                    with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
                                        inner_zip.extractall(dataset_path)
                                    os.remove(inner_zip_path)

                                data_train_dir = os.path.join(dataset_path, "data.yaml")

                                if os.path.exists(data_train_dir):
                                    with open(data_train_dir, 'r') as file:
                                        data_yaml = yaml.safe_load(file)
                                    
                                    # Thay thế các đường dẫn
                                    data_yaml['train'] = os.path.join('train', 'images')
                                    data_yaml['val'] = os.path.join('val', 'images')
                                    data_yaml['test'] = os.path.join('test', 'images')
                                else:
                                    data_train_dir = os.path.join(dataset_path, "dataset.yaml")
                                    with open(data_train_dir, 'r') as file:
                                        data_yaml = yaml.safe_load(file)

                                # Ghi lại data.yaml
                                with open(data_train_dir, 'w') as file:
                                    yaml.dump(data_yaml, file, default_flow_style=False, sort_keys=False)
                    try:
                        files = [os.path.join(weight_path, filename) for filename in os.listdir(weight_path) if os.path.isfile(os.path.join(weight_path, filename))]
                    except:
                        pass
                        
                    if len(files) > 0:
                        model = YOLO(files[0])
                    else:
                        model = YOLO("yolo12n.pt")

                    train_dir = os.path.join(os.getcwd(),f"{project_id}")
                    print(data_train_dir, train_dir)

                    model.train(data=data_train_dir, imgsz=imgsz, epochs=epochs, project=train_dir)
                    # run_train(model, data_train_dir, imgsz, epochs, train_dir)
                    
                    # subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                    # latest_subdir = max(subdirs, key=os.path.getmtime)
                    # # checkpoint_model = f'{latest_subdir}/weights/last.pt'
                        
                    # # best_model = f'{latest_subdir}/weights/best.pt'
                    # if os.path.exists(train_dir):
                    #     import datetime
                    #     # checkpoint_name = upload_checkpoint(checkpoint_model, project_id, token)
                    #     # if checkpoint_name:
                    #     #     weight_path_final = os.path.join(clone_dir, "models", checkpoint_name)
                    #     #     os.makedirs(weight_path_final, exist_ok=True)
                    #     #     shutil.copy(checkpoint_model, weight_path_final)

                    #     # output_dir = "./data/checkpoint"
                    #     now = datetime.datetime.now()
                    #     date_str = now.strftime("%Y%m%d")
                    #     time_str = now.strftime("%H%M%S")
                    #     version = f'{date_str}-{time_str}'

                    #     upload_checkpoint(project, version, train_dir)

                    # if push_to_hub:
                    repo = create_repo(repo_id=hf_model_id, private=False, token=push_to_hub_token,exist_ok=True)
                                            # Đăng nhập vào Hugging Face
                    login(token=push_to_hub_token)

                    # Đẩy thư mục huấn luyện lên Hugging Face Hub
                    upload_folder(
                        folder_path=f'{train_dir}/train',  # Thư mục chứa kết quả huấn luyện
                        repo_id=repo.repo_id,  # ID của mô hình trên Hugging Face
                        token=push_to_hub_token
                    )

                    CHANNEL_STATUS[channel_name]["status"] = "Done"

                # func_train_model(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id)
                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, imgsz, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id))

                train_thread.start()

                return {"message": "train completed successfully"}
            
            except Exception as e:
                print(e)
                return {"message": f"train failed: {e}"}
            # try:
            #     checkpoint = kwargs.get("checkpoint")
            #     if checkpoint:
            #         model = YOLO(f"checkpoints/uploads/{checkpoint}")
            #     else:
            #         model = YOLO("yolov8n.pt")

            #     epochs = kwargs.get("epochs", 2)
            #     imgsz = kwargs.get("imgsz", 640)
            #     data = kwargs.get("data", "coco8.yaml")
            #     # check if folder named project exists
            #     if not os.path.exists(f"./yolov8/{project}"):
            #         os.makedirs(f"./yolov8/{project}")
            #     result = model.train(data=data, imgsz=imgsz, epochs=epochs, project=f"./yolov8/{project}")
            #     return {"message": "train completed successfully"}
            # except Exception as e:
            #     return {"message": f"train failed: {e}"}
        
        elif command.lower() == "tensorboard":
            project_id = kwargs.get("project_id")
            clone_dir = os.path.join(os.getcwd())
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                if project_id:
                    p = subprocess.Popen(f"tensorboard --logdir ./{project_id} --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                else:
                    p = subprocess.Popen(f"tensorboard --logdir {clone_dir}/ --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        elif command.lower() == "dashboard":
            # link = promethus_grafana.generate_link_public("ml_00")
            return {"Share_url": ""}
        
        elif command.lower() == "predict":
                image_64 = kwargs.get("image",None)
                if not image_64:
                    image_64 = kwargs.get('data', {}).get('image')
                    
                confidence_threshold = kwargs.get("confidence_threshold", 0.2)
                # data = kwargs.get("data", None)
                model_type = kwargs.get("model_type", None)
                if not model_type:
                    model_type = kwargs.get('data', 'polygonlabels').get('type')

                if model_type == "" or not model_type:
                    model_type = "polygonlabels"

                print(model_type)

                results = []
                if len(image_64)>0:
                    img = decode_base64_to_image(image_64)
                else:
                    return {"message": "predict failed", "result": None}
               
                # Giải mã base64 thành bytes
                image_data = base64.b64decode(image_64)
                # Đọc ảnh từ bytes
                image = Image.open(BytesIO(image_data))
                width, height = image.size

                # image.save('output_image.png')
                # width, height = img.shape[1], img.shape[0]
                # if kwargs.get("checkpoint_version") and kwargs.get("checkpoint"):
                #     checkpoint_version = kwargs.get("checkpoint_version")
                #     checkpoint_id = kwargs.get("checkpoint")
                #     weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                #     if not os.path.exists(weight_path):
                #         weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                #         checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                #         if checkpoint_name:
                #             # weight_path = os.path.join(clone_dir, f"models/{checkpoint_name}")
                #             # if not os.path.exists(weight_path):
                #             with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                #                 zip_ref.extractall(weight_path)

                #     model = YOLO(weight_path)
                # else:
                if model_type == "rectanglelabels":
                    model = YOLO("yolov8n.pt")
                elif model_type == "polygonlabels":
                    model = YOLO("yolov8n-seg.pt")
                else:
                    # print("yolov8n-seg")
                    model = YOLO("yolov8n-seg.pt")
            
                result = model(img)
                
                if model_type == "rectanglelabels":
                    print(result)
                    boxes = result[0].boxes
                    names = result[0].names
                    for i in range(len(boxes)):
                        x_left, y_top, x_right, y_right = boxes.xyxy[i].tolist()
                        predicted_label_idx = boxes.cls[i].item()  # Chuyển đổi nhãn về dạng số
                        predicted_label = names[int(predicted_label_idx)]
                        avg_score = boxes.conf[i].item()  # Điểm tin cậy của dự đoán
                        if avg_score < float(confidence_threshold):
                            continue

                        # Thêm thông tin vào danh sách kết quả
                        results.append({
                            "type": 'rectanglelabels',
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "x": (float(x_left)/width)*100,
                                "y": (float(y_top)/height)*100,
                                "width": ((x_right-x_left)/width)*100,
                                "height": ((y_right-y_top)/height)*100,
                                "rotation": 0,
                                "rectanglelabels": [predicted_label]  # Chắc chắn rằng đây là danh sách
                            },
                            "score": avg_score
                        })
                   
                elif model_type == "polygonlabels":
                    for i in range(len(result[0].masks.xy)):  # Sử dụng thuộc tính xy để lấy tọa độ
                        # Lấy thông tin về các segment (phân đoạn)
                        segments = result[0].masks.xy[i]  # Lấy các tọa độ phân đoạn trong pixel
                        # # if not data:
                        # polygons, has_holes = mask_to_polygons(segments, width, height)
                        # else:
                        polygons = [[float(x), float(y)] for x, y in segments]

                        # Điều chỉnh tỷ lệ cho đúng kích thước ảnh
                        # Chia mỗi tọa độ x, y cho width và height để đưa về tỷ lệ phần trăm
                        polygons = [[x / width * 100, y / height * 100] for x, y in polygons]

                        label_index = result[0].boxes.cls[i].item()  # Lấy chỉ số của nhãn dự đoán
                        avg_score = result[0].boxes.conf[i].item()  # Lấy điểm tin cậy của nhãn dự đoán
                        predicted_label = result[0].names[label_index]  # Tên của nhãn (dựa trên index)

                        if avg_score < float(confidence_threshold):
                            continue
                        # Chuyển đổi các điểm segment thành float

                        if polygons:  # Nếu có điểm thì mới thêm vào kết quả
                            results.append({
                                "type": 'polygonlabels',
                                "original_width": width,
                                "original_height": height,
                                "image_rotation": 0,
                                "value": {
                                    "points": polygons,  # Danh sách các điểm (x, y) tạo thành polygon
                                    "polygonlabels": [predicted_label]  # Nhãn phân đoạn
                                },
                                "score": avg_score
                            })
                
                    results = nms_for_polygons(results)

                print(results)
                return {"message": "predict completed successfully",
                        "result": results}

            # except:
            #     return {"message": "predict failed", "result": None}
        
        elif command.lower() == "toolbar":
             # try:
                image_64 = kwargs.get("image")
                data = kwargs.get("data", None)
                model_type = kwargs.get("model_type")
                results = []
                img = Image.open(io.BytesIO(
                base64.b64decode(image_64))).convert('RGB')
                img_width, img_height = img.size 
                width, height = img.shape[1], img.shape[0]
                if kwargs.get("checkpoint_version") and kwargs.get("checkpoint"):
                    checkpoint_version = kwargs.get("checkpoint_version")
                    checkpoint_id = kwargs.get("checkpoint")
                    weight_path = os.path.join(clone_dir, f"models/{checkpoint_version}")
                    if not os.path.exists(weight_path):
                        weight_zip_path = os.path.join(clone_dir, "data_zip/weights.zip")
                        checkpoint_name = download_checkpoint(weight_zip_path, project_id, checkpoint_id, token)
                        if checkpoint_name:
                            # weight_path = os.path.join(clone_dir, f"models/{checkpoint_name}")
                            # if not os.path.exists(weight_path):
                            with zipfile.ZipFile(weight_zip_path, 'r') as zip_ref:
                                zip_ref.extractall(weight_path)

                    model = YOLO(weight_path)
                else:
                    
                    if model_type == "rectanglelabels":
                        model = YOLO("yolov8n.pt")
                    elif model_type == "polygonlabels":
                        model = YOLO("yolov8n-seg.pt")

                result = model(img)
                
                if model_type == "rectanglelabels":
                    # print(result)
                    boxes = result[0].boxes
                    names = result[0].names
                    for i in range(len(boxes)):
                        x_left, y_top, w, h = boxes.xywh[i].tolist()
                        predicted_label_idx = boxes.cls[i].item()  # Chuyển đổi nhãn về dạng số
                        predicted_label = names[int(predicted_label_idx)]
                        avg_score = boxes.conf[i].item()  # Điểm tin cậy của dự đoán

                        # Thêm thông tin vào danh sách kết quả
                        results.append({
                            "type": 'rectanglelabels',
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "x": float(x_left)/img_width*100,
                                "y": float(y_top)/img_height*100,
                                "width": float(w)/img_width*100,
                                "height": float(h)/img_height*100,
                                "rotation": 0,
                                "rectanglelabels": [predicted_label]  # Chắc chắn rằng đây là danh sách
                            },
                            "score": avg_score
                        })
                   
                elif model_type == "polygonlabels":
                    for i in range(len(result[0].masks.xy)):  # Sử dụng thuộc tính xy để lấy tọa độ
                        # Lấy thông tin về các segment (phân đoạn)
                        segments = result[0].masks.xy[i]  # Lấy các tọa độ phân đoạn trong pixel
                        if not data:
                            polygons, has_holes = mask_to_polygons(segments, width, height)
                        else:
                            polygons = [[float(x), float(y)] for x, y in segments]

                        label_index = result[0].boxes.cls[i].item()  # Lấy chỉ số của nhãn dự đoán
                        avg_score = result[0].boxes.conf[i].item()  # Lấy điểm tin cậy của nhãn dự đoán
                        predicted_label = result[0].names[label_index]  # Tên của nhãn (dựa trên index)

                        # Chuyển đổi các điểm segment thành float

                        if polygons:  # Nếu có điểm thì mới thêm vào kết quả
                            results.append({
                                "type": 'polygonlabels',
                                "original_width": width,
                                "original_height": height,
                                "image_rotation": 0,
                                "value": {
                                    "points": polygons,  # Danh sách các điểm (x, y) tạo thành polygon
                                    "polygonlabels": [predicted_label]  # Nhãn phân đoạn
                                },
                                "score": avg_score
                            })

                print(results)
                return {"message": "predict completed successfully",
                        "result": results}

            # except:
            #     return {"message": "predict failed", "result": None}
        
        elif command == "status":
            channel = kwargs.get("channel", None)
            
            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}
                
                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})
                
                return {"channels": channels}  
        # elif command.lower() == "logs":
        #     logs = fetch_logs()
        #     return {"message": "command not supported", "result": logs}
        
        else:
            return {"message": "command not supported", "result": None}

            # return {"message": "train completed successfully"}

    def model(self, **kwargs):
        print(kwargs)
        task = kwargs.get("task", "bounding-boxes-segmentation")
        
        import gradio as gr
        if task:
            TYPE_ENV = task

        # css = """
        # .feedback .tab-nav {
        #     justify-content: center;
        # }

        # .feedback button.selected{
        #     background-color:rgb(115,0,254); !important;
        #     color: #ffff !important;
        # }

        # .feedback button{
        #     font-size: 16px !important;
        #     color: black !important;
        #     border-radius: 12px !important;
        #     display: block !important;
        #     margin-right: 17px !important;
        #     border: 1px solid var(--border-color-primary);
        # }

        # .feedback div {
        #     border: none !important;
        #     justify-content: center;
        #     margin-bottom: 5px;
        # }

        # .feedback .panel{
        #     background: none !important;
        # }


        # .feedback .unpadded_box{
        #     border-style: groove !important;
        #     width: 500px;
        #     height: 345px;
        #     margin: auto;
        # }

        # .feedback .secondary{
        #     background: rgb(225,0,170);
        #     color: #ffff !important;
        # }

        # .feedback .primary{
        #     background: rgb(115,0,254);
        #     color: #ffff !important;
        # }

        # .upload_image button{
        #     border: 1px var(--border-color-primary) !important;
        # }
        # .upload_image {
        #     align-items: center !important;
        #     justify-content: center !important;
        #     border-style: dashed !important;
        #     width: 500px;
        #     height: 345px;
        #     padding: 10px 10px 10px 10px
        # }
        # .upload_image .wrap{
        #     align-items: center !important;
        #     justify-content: center !important;
        #     border-style: dashed !important;
        #     width: 500px;
        #     height: 345px;
        #     padding: 10px 10px 10px 10px
        # }

        # .webcam_style .wrap{
        #     border: none !important;
        #     align-items: center !important;
        #     justify-content: center !important;
        #     height: 345px;
        # }

        # .webcam_style .feedback button{
        #     border: none !important;
        #     height: 345px;
        # }

        # .webcam_style .unpadded_box {
        #     all: unset !important;
        # }

        # .btn-custom {
        #     background: rgb(0,0,0) !important;
        #     color: #ffff !important;
        #     width: 200px;
        # }

        # .title1 {
        #     margin-right: 90px !important;
        # }

        # .title1 block{
        #     margin-right: 90px !important;
        # }

        # """

        with gr.Blocks(css=css) as demo2:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np
            def numpy_to_base64(np_array, format="JPEG"):
                    """
                    Chuyển đổi một mảng NumPy (hình ảnh) sang chuỗi base64.
                    
                    Parameters:
                    - np_array: Mảng NumPy chứa dữ liệu hình ảnh.
                    - format: Định dạng tệp hình ảnh (ví dụ: JPEG, PNG).
                    
                    Returns:
                    - Chuỗi base64 đại diện cho hình ảnh.
                    """
                    # Chuyển đổi mảng NumPy thành đối tượng hình ảnh PIL
                    image = Image.fromarray(np_array.astype("uint8"))
                    
                    # Lưu hình ảnh vào một buffer bằng BytesIO
                    buffered = BytesIO()
                    image.save(buffered, format=format)
                    
                    # Mã hóa nội dung buffer thành base64
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    return img_base64
            
            def predict(input_img):
                import cv2
                print(input_img)
                img_64 = numpy_to_base64(input_img)
                result = self.action("predict", data={"image": img_64})
                print(result)
                if result['result']:
                    if TYPE_ENV == "DETECTION" or TYPE_ENV == 'polygon-segmentation':
                        for res in result['result']:  # Lặp qua từng phần tử trong danh sách kết quả
                    # points = res['value']['points']  # Tọa độ box [x_left, y_top, x_right, y_bottom]
                            original_width = res['original_width']
                            original_height = res['original_height']
                            label = res['value']['rectanglelabels'][0]  # Tên nhãn (label)
                            score = res['score']  # Điểm tin cậy (confidence score)

                            x_left_pct = res['value']['x']
                            y_top_pct = res['value']['y']
                            width_pct = res['value']['width']
                            height_pct = res['value']['height']

                            # Tính toán lại tọa độ và kích thước bounding box trong ảnh gốc
                            x_left = (x_left_pct / 100) * original_width
                            y_top = (y_top_pct / 100) * original_height
                            width = (width_pct / 100) * original_width
                            height = (height_pct / 100) * original_height

                            # Chuyển đổi tọa độ thành kiểu int nếu cần
                            x_left, y_top, x_right, y_bottom = int(x_left), int(y_top), int(x_left + width), int(y_top + height)

                            # Vẽ hình chữ nhật lên ảnh
                            input_img = cv2.rectangle(input_img, (x_left, y_top), (x_right, y_bottom), color=(255, 0, 0), thickness=2)

                            # Vẽ tên nhãn lên ảnh
                            input_img = cv2.putText(input_img, label, (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            score_text = f"Score: {score:.2f}"
                            input_img = cv2.putText(input_img, score_text, (x_left, y_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                            
                    elif TYPE_ENV == "SEGMENT" or TYPE_ENV == 'bounding-boxes-segmentation':
                        for res in result['result']:
                            original_width = res['original_width']
                            original_height = res['original_height']
                            points = np.array(res['value']['points'], np.int32)  # Chuyển các điểm về dạng numpy array
                            points = points * [original_width / 100, original_height / 100]
                            label = res['value']['polygonlabels'][0] 

                            score = res['score']  
                            points = points.astype(np.int32)

                            input_img = cv2.fillPoly(input_img, [points], color=(0, 255, 0))
                            input_img = cv2.polylines(input_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                            input_img = cv2.putText(input_img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            score_text = f"Score: {score:.2f}"
                            input_img = cv2.putText(input_img, score_text, (points[0][0], points[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                return input_img

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os
                checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                                   checkpoint_list]
                if os.path.exists(f"yolov8/{project}"):
                    for folder in os.listdir(f"yolov8/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in
                                                       os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                       i.endswith(".pt")]
                            project_checkpoint_list = [
                                f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                                for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)

                return "<br>".join(checkpoint_list)

            def process_video(video_path):
                # Mở video tải lên từ đường dẫn
                cap = cv2.VideoCapture(video_path)

                # Kiểm tra nếu video được mở thành công
                if not cap.isOpened():
                    return "Error: Could not open video"

                # Lấy thông tin về video
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Tạo đối tượng để ghi video đầu ra
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (frame_width, frame_height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # Dừng nếu không còn frame nào

                    # Xử lý frame
                    processed_frame = predict(frame)

                    # Viết frame đã xử lý vào video output
                    out.write(processed_frame)

                # Giải phóng các tài nguyên
                cap.release()
                out.release()

                return 'processed_video.mp4'

            def clear():
                """Hàm xóa đầu vào và đầu ra."""
                return None, None
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Demo", id=0):
                    with gr.Tabs() as demo_sub_tabs:
                            # Tab con cho Upload Image
                            with gr.TabItem("Upload Image"):
                                with gr.Row():
                                    gr.Markdown("## Input", elem_classes=["title1"])
                                    gr.Markdown("## Output", elem_classes=["title1"])

                                gr.Interface(predict,
                                                gr.Image(elem_classes=["upload_image"], sources="upload", container=False, height=345,
                                                        show_label=False),
                                                gr.Image(elem_classes=["upload_image"], container=False, height=345, show_label=False),
                                                # allow_flagging=False
                                                )

                            with gr.TabItem("Webcam Capture"):
                                with gr.Row():
                                    gr.Markdown("## Input", elem_classes=["title1"])
                                    gr.Markdown("## Output", elem_classes=["title1"])

                                # Row for Input and Output columns
                                with gr.Row():
                                    # Column for Input (Webcam and Buttons)
                                    with gr.Column():
                                        # Webcam stream input
                                        webcam_feed = gr.Image(
                                            sources="webcam",
                                            elem_classes=["upload_image"],
                                            container=False,
                                            height=345,
                                            show_label=False
                                        )

                                        # Buttons under the Input column
                                        with gr.Row():
                                            clear_button = gr.Button("Clear")
                                            submit_button = gr.Button("Submit", elem_classes=["primary"])


                                    # Column for Output
                                    with gr.Column():
                                        # Processed Output
                                        processed_output = gr.Image(
                                            elem_classes=["upload_image"],
                                            container=False,
                                            height=345,
                                            show_label=False
                                        )

                                # Button functionality
                                submit_button.click(
                                    fn=predict,  # Gọi hàm xử lý
                                    inputs=webcam_feed,  # Đầu vào là webcam
                                    outputs=processed_output  # Hiển thị kết quả
                                )

                                clear_button.click(
                                    fn=clear,  # Gọi hàm xóa
                                    inputs=None,  # Không cần đầu vào
                                    outputs=[webcam_feed, processed_output]  # Xóa cả webcam feed và kết quả
                                )

                with gr.TabItem("Video", id=1):
                    # gr.Image(elem_classes=["upload_image"], sources="clipboard", height=345, container=False,
                    #           show_label=False)
                    with gr.Tabs() as demo_sub_tabs:
                        with gr.TabItem("Live cam"):
                            with gr.Column(elem_classes=["my-column"]):
                                with gr.Group(elem_classes=["my-group"]):
                                    image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)

                                image.stream(
                                    fn=predict, inputs=[image], outputs=[image], time_limit=10
                                )
                        with gr.TabItem("Upload Video"):
                            with gr.Row():
                                gr.Markdown("## Input", elem_classes=["title1"])
                                gr.Markdown("## Output", elem_classes=["title1"])

                            with gr.Row():
                                # Cột đầu tiên cho video input
                                with gr.Column():
                                    video_input = gr.Video(label="Upload Video", format="mp4")
                                    submit_button = gr.Button("Process Video")

                                # Cột thứ hai cho video output và nút submit
                                with gr.Column():
                                    video_output = gr.Video(label="Processed Video")
                                
                                submit_button.click(
                                        fn=process_video, inputs=[video_input], outputs=[video_output]
                                    )


        gradio_app, local_url, share_url = demo2.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)

        return {"share_url": share_url, 'local_url': local_url}

    def model_trail(self, **kwargs):
        while self.is_init_model_trial:
            time.sleep(1)

        if len(self.model_trial_local_url) > 0 and len(self.model_trial_public_url) > 0:
            return {"share_url": self.model_trial_public_url, 'local_url': self.model_trial_local_url}

        self.is_init_model_trial = True
        print(kwargs)
        import gradio as gr

        def mt_predict(input_img):
            result = self.action("predict", data={"img": input_img, "type": TYPE_ENV})

            if result['result']:
                if TYPE_ENV == "DETECTION":
                    for res in result['result']:  # Lặp qua từng phần tử trong danh sách kết quả
                        points = res['value']['points']  # Tọa độ box [x_left, y_top, x_right, y_bottom]
                        label = res['value']['rectanglelabels'][0]  # Tên nhãn (label)
                        score = res['score']  # Điểm tin cậy (confidence score)

                        # Chuyển đổi tọa độ thành int
                        x_left, y_top, x_right, y_bottom = [int(coord) for coord in points]

                        # Vẽ hình chữ nhật lên ảnh
                        input_img = cv2.rectangle(input_img, (x_left, y_top), (x_right, y_bottom), color=(255, 0, 0), thickness=2)

                        # Vẽ tên nhãn lên ảnh
                        input_img = cv2.putText(input_img, label, (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif TYPE_ENV == "SEGMENT":
                    for res in result['result']:
                        points = np.array(res['value']['points'], np.int32)  # Chuyển các điểm về dạng numpy array
                        label = res['value']['polygonlabels'][0]  # Nhãn của đối tượng
                        score = res['score']  # Điểm tin cậy của dự đoán
                        
                        # Vẽ mask lên ảnh
                        input_img = cv2.fillPoly(input_img, [points], color=(0, 255, 0))  # Màu xanh cho vùng phân đoạn
                        
                        # Vẽ tên nhãn lên ảnh (tại điểm đầu tiên của polygon)
                        x_text, y_text = points[0][0], points[0][1]
                        input_img = cv2.putText(input_img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return input_img

        def mt_trial_training(dataset_choosen):
            print(f"Training with {dataset_choosen}")
            result = self.action("1", "train", collection="", data=dataset_choosen)
            return result['message']

        def mt_download_btn(evt: gr.SelectData):
            print(f"Downloading {dataset_choosen}")
            return f'<a href="/yolov8/datasets/{evt.value}" class="aixblock__download-button"><i class="fa fa-download"></i> Download</a>'

        def mt_get_checkpoint_list(project):
            print("GETTING CHECKPOINT LIST")
            print(f"Proejct: {project}")
            import os
            checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
            checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                               checkpoint_list]
            if os.path.exists(f"yolov8/{project}"):
                for folder in os.listdir(f"yolov8/{project}"):
                    if "train" in folder:
                        project_checkpoint_list = [i for i in
                                                   os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                   i.endswith(".pt")]
                        project_checkpoint_list = [
                            f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                            for i in project_checkpoint_list]
                        checkpoint_list.extend(project_checkpoint_list)

            return "<br>".join(checkpoint_list)

        def mt_tab_changed(tab):
            if tab == "Download":
                mt_get_checkpoint_list(project="1")

        def mt_upload_file(file):
            return "File uploaded!"

        with gr.Blocks(css=css, js=js) as demo:
            gr.Markdown("AIxBlock", elem_classes=["aixblock__title"])

            with gr.Tabs(elem_classes=["aixblock__tabs"]):
                with gr.TabItem("Demo", id=0):
                    # with gr.Row():
                    #     gr.Markdown("Input", elem_classes=["aixblock__input-title"])
                    #     gr.Markdown("Output", elem_classes=["aixblock__output-title"])

                    gr.Interface(
                        mt_predict,
                        gr.Image(elem_classes=["aixblock__input-image"], container=False, height=345),
                        gr.Image(elem_classes=["aixblock__output-image"], container=False, height=345),
                        allow_flagging="never",
                    )

                # with gr.TabItem("Webcam", id=1):
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("Dataset template to prepare your own and initiate training")
                    # with gr.Row():
                    # get all filename in datasets folder
                    datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                    dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True,
                                                  type="value")
                    # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                    download_link = gr.HTML("""<em>Please select a dataset to download</em>""")
                    dataset_choosen.select(mt_download_btn, None, download_link)

                    # when the button is clicked, download the dataset from dropdown
                    # download_btn
                    gr.Markdown("Upload your sample dataset to have a trial training")
                    # gr.File(file_types=['tar','zip'])

                    gr.Interface(
                        mt_predict,
                        gr.File(elem_classes=["aixblock__input-image"], file_types=['tar', 'zip'], container=False),
                        gr.Label(elem_classes=["aixblock__output-image"], container=False),
                        allow_flagging="never",
                    )

                    with gr.Column(elem_classes=["aixblock__trial-train"]):
                        gr.Button("Trial Train", variant="primary").click(mt_trial_training, dataset_choosen, None)
                        gr.HTML(value=f"<em>You can attemp up to {2} FLOps</em>")

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True,
                                                       server_name='0.0.0.0', show_error=True)
        self.model_trial_public_url = share_url
        self.model_trial_local_url = local_url
        self.is_init_model_trial = False

        return {"share_url": share_url, 'local_url': local_url}

    def download(self, project, **kwargs):
        return super(self).download(project, **kwargs)
