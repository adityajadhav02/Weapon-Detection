import cv2
import numpy as np
import os
import time

class WeaponDetector:
    def __init__(self, weights_path=None, config_path=None):
        """
        Initialize the weapon detector with YOLO model
        
        Args:
            weights_path: Path to the YOLO weights file
            config_path: Path to the YOLO configuration file
        """
        # Default paths if not provided
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov3_training_2000.weights")
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov3_testing.cfg")
        
        # Check if files exist
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        # Load the YOLO model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Define classes
        self.classes = ["Weapon"]
        
        # Get output layer names
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()
        
        # Generate random colors for visualization
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        print("Weapon detector initialized successfully")
    
    def detect_image(self, image_path):
        """
        Detect weapons in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection results with bounding boxes and confidence scores
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Get image dimensions
        height, width, _ = img.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Forward pass
        outs = self.net.forward(self.output_layer_names)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Prepare results
        results = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                results.append({
                    'class': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': [x, y, w, h]
                })
        
        return results
    
    def process_frame(self, frame):
        """
        Process a video frame for weapon detection
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Processed frame with detection boxes
        """
        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Forward pass
        outs = self.net.forward(self.output_layer_names)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Draw detection boxes
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), font, 2, color, 2)
        
        return frame
    
    def detect_video(self, video_path, output_path=None):
        """
        Process a video file for weapon detection
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the processed video (optional)
            
        Returns:
            List of frames with detections
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        processed_frames = []
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            processed_frames.append(processed_frame)
            
            # Write to output video if writer is initialized
            if writer:
                writer.write(processed_frame)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time
                print(f"Processed {frame_count} frames at {fps_processing:.2f} fps")
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        
        print(f"Video processing completed. Processed {frame_count} frames.")
        return processed_frames 