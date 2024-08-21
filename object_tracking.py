





from ultralytics import YOLO
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

class Object_tracking:
    Detection_Model=None
    Encoding_Mode=None
    tracker = None
    tracks = None
    detections=None
    def __init__(self,Detection_Model_Path,Encoding_Mode_Path) -> None:
        max_cosine_distance = 0.4
        nn_budget = None
        if "yolo" in Detection_Model_Path:
            self.Detection_Model=YOLO(Detection_Model_Path)
        else:
            self.Detection_Model=None
        self.Encoding_Mode= generate_detections.create_box_encoder(Encoding_Mode_Path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)

    def detect(self,frame,detection_threshold:float=0.5,detect_class:list=["all"])->list:
        results = self.Detection_Model(frame)
        detections = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if "all" not in detect_class :                
                    if self.Detection_Model.names[class_id] not in detect_class:
                            continue
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score,self.Detection_Model.names[class_id]])
        self.detections=detections.copy()
        return detections.copy()

    def track(self, frame, detections=None):
        if detections==None:
            detections=self.detections.copy()
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return
        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]
        features = self.Encoding_Mode(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            tracks.append(Track(id, bbox))
        self.tracks = tracks

class Track:
    track_id = None
    bbox = None
    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox