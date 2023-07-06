import cv2
import numpy as np


class YoloDetector:
    def __init__(self, model_cfg, model_weights, classes):
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("GPU Backend selected")
        self.classes = classes
        self.layer_names = self.net.getLayerNames()
        self.outputlayers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    @staticmethod
    def get_output_format(box):
        x, y, w, h = box
        return int(x), int(y), int(x+w), int(y+h)

    # the output format is output = {class:[positions]}
    def detect(self, img, conf=0.2, nms_thresh=0.3, non_max_suppression=True, class_conf=None):
        if class_conf is None:
            class_conf = []
        if len(class_conf) < len(self.classes):
            conf = [conf] * len(self.classes)
        else:
            conf = class_conf
        class_conf_dict = {k: conf[i] for i, k in enumerate(self.classes)}
        final_result = {k: [] for k in self.classes}
        confidences = {k: [] for k in self.classes}
        boxes = {k: [] for k in self.classes}
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.outputlayers)
        Height, Width, _ = img.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf[int(class_id)]:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - (w / 2)
                    y = center_y - (h / 2)
                    confidences[self.classes[class_id]].append(float(confidence))
                    boxes[self.classes[class_id]].append([int(i) for i in [x, y, w, h]])
        indices = {}
        if non_max_suppression:
            for class_name, box in boxes.items():
                indices[class_name] = cv2.dnn.NMSBoxes(box, confidences[class_name], class_conf_dict[class_name], nms_thresh)
        else:
            for class_name, box in boxes.items():
                indices[class_name] = [[w] for w in range(len(box))]
        for key, index in indices.items():
            for i in index:
                select = i[0]
                final_result[key].append(self.get_output_format(boxes[key][select]))
            
        return final_result
