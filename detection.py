
from imageai.Detection import VideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

def intersection_over_union(current_box_points, previous_box_points):
#https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
#resnet50_coco_best_v2.0.1.h5

    # determine the (x, y)-coordinates of the intersection rectangle
	x1 = max(previous_box_points[0], current_box_points[0])
	y1 = max(previous_box_points[1], current_box_points[1])
	x2 = min(previous_box_points[2], current_box_points[2])
	y2 = min(previous_box_points[3], current_box_points[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	previous_box_area = (previous_box_points[2] - previous_box_points[0] + 1) * (previous_box_points[3] - previous_box_points[1] + 1)
	current_box_area = (current_box_points[2] - current_box_points[0] + 1) * (current_box_points[3] - current_box_points[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	result_iou = interArea / float(previous_box_area + current_box_area - interArea)
 
	# return the intersection over union value
	return result_iou


def forFrame(frame_number, output_array, output_count, returned_frame):
    temp_detections = []
    for id, detection in enumerate(output_array):
        #get current_box
        current_box_points = detection.get('box_points')
        current_box_name = detection.get('name')
        x1 = current_box_points[0]
        y1 = current_box_points[1]

        iou_id_all=[]
        iou_result_all=[]
        
        #get previous_box
        for previous_box in CameraDetector.detections:
            previous_box_points = previous_box.get('box_points')
            previous_box_id = previous_box.get('id')
            previous_box_name = previous_box.get('name')

            #IOU calculation
            iou_result = intersection_over_union(current_box_points, previous_box_points)

            #IOU index can be higher depends of the camera's quality
            if iou_result >0.3:
                iou_result_all.append(iou_result)
                iou_id_all.append(previous_box_id)

        if iou_result_all:
            id = iou_id_all[np.argmax(iou_result_all)] 
        #if the object was not detected before on the scene, new id is assign       
        else:
            CameraDetector.id += 1
            id = CameraDetector.id

        test = {'id':id, 'box_points': current_box_points,'name': current_box_name}
        cv2.putText(returned_frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255),2)
        temp_detections.append(test)
    
    CameraDetector.detections = temp_detections

    cv2.imshow("Detection", returned_frame)
    cv2.waitKey(1)
    #print(output_array)

class CameraDetector:

    detections = []
    id = 0

    def __init__(self, cam):

        self.execution_path = os.getcwd()

        self.color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}
        
        self.resized = False

        self.cap = cv2.VideoCapture(cam)

    def start(self):

        detector = VideoObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(self.execution_path , 
                                            "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detector.detectObjectsFromVideo(camera_input=self.cap, 
                                        output_file_path=os.path.join(self.execution_path, 
                                                                     "video_frame_analysis") , 
                                        frames_per_second=30, 
                                        per_frame_function=forFrame,  
                                        minimum_percentage_probability=70, 
                                        return_detected_frame=True)




if __name__=="__main__":
    cameraDetector = CameraDetector(0)
    cameraDetector.start()



