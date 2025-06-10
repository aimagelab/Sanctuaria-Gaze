import numpy as np
import torch

class Detectron2:

    def __init__(self):
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo

        # Configure Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.model = DefaultPredictor(cfg)

    def detect(self, image):
        # Perform object detection
        outputs = self.model(np.array(image))
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        class_names = self.model.metadata.get("thing_classes", [])
        return boxes, classes, class_names

class YOLOv11:
    def __init__(self):
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        self.model = YOLO("yolo11m")
        
    def detect(self, image):
        yolo_results = self.model.predict(image)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_results[0].boxes.conf.cpu().numpy() 
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = self.model.names  # Get class names
        return boxes, classes, class_names

class YOLOv8_World:

    def __init__(self, class_file="object_classes.txt"):
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        self.model = YOLO("yolov8x-worldv2.pt")  # or choose yolov8m/l-world.pt

        # Load custom classes from file
        with open(class_file, "r") as f:
            class_list = [line.strip() for line in f if line.strip()]
        self.model.set_classes(class_list)
        
    def detect(self, image):
        # Perform object detection with YOLO-World
        yolo_results = self.model.predict(image)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_results[0].boxes.conf.cpu().numpy() 
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = self.model.names  # Get class names
        
        return boxes, classes, class_names

class OWLv2:
    def __init__(self, class_file="object_classes.txt"):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        # Load object names from file
        with open(class_file, "r") as f:
            self.texts = [[line.strip() for line in f if line.strip() and not line.startswith("#")]]
    
    def detect(self, image):
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.6)[0]
        
        return results['boxes'], results['labels'], self.texts

class SAM2:
    def __init__(self):
        from ultralytics import SAM
        
        self.model = SAM("sam2.1_b.pt")

    def predict(self, prompt_point, prompt_label, image):
        # Perform segmentation
        masks = self.model.predict(
            image,
            points=prompt_point,
            labels=prompt_label
        )

        best_mask = masks[0].masks.data[0].cpu().numpy()

        return best_mask

class LightGlue:
    def __init__(self, reference_image_path, threshold=50, debug=False):
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image

        # Initialize LoFTR model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        self.threshold = threshold # Set the threshold for the number of matches
        self.debug = debug

        # Load and store reference image
        self.reference_img = load_image(reference_image_path).to(self.device)
        

    def extract_bounding_box(self, points):
        # Convert to CPU and NumPy
        points = points.cpu().numpy()
        
        # Compute bounding box
        minx, miny = points.min(axis=0)
        maxx, maxy = points.max(axis=0)
        
        # Define bounding box corners
        bbox = [[minx, miny, maxx, maxy]]
        
        return bbox

    def detect(self, image):
        from lightglue.utils import numpy_image_to_torch, rbd
        # Prepare query image
        query_tensor = numpy_image_to_torch(image).to(self.device)

        # extract local features
        feats0 = self.extractor.extract(self.reference_img)
        feats1 = self.extractor.extract(query_tensor)

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
            
        print(f"Found {len(m_kpts0)} matches.")

        if self.debug:
            # Visualize the matches
            from lightglue.viz2d import plot_images, plot_matches, save_plot
            
            axes = plot_images([self.reference_img, query_tensor])
            plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            save_plot('feature_matches.png')

        if len(m_kpts0) < self.threshold:
            return [], [], []

        boxes = self.extract_bounding_box(m_kpts1)

        classes = np.zeros(len(boxes), dtype=int)  # All points are same class
        class_names = ["match"]
        
        return boxes, classes, class_names