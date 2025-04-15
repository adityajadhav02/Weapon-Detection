import cv2

class Detector:
    def __init__(self, model):
        self.model = model

    def process_frame(self, frame):
        """Process a single frame for weapon detection.
        
        Args:
            frame: numpy array of the frame to process
            
        Returns:
            Processed frame with detection boxes drawn
        """
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Convert frame to RGB for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(rgb_frame)
        
        # Draw detection boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def detect_weapons(self, image):
        """Detect weapons in an image.
        
        Args:
            image: numpy array of the image to process
            
        Returns:
            List of detection results with confidence and bounding boxes
        """
        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(rgb_image)
        
        # Format results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to [x, y, w, h] format
                w = x2 - x1
                h = y2 - y1
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                detections.append({
                    'confidence': conf,
                    'box': [float(x1), float(y1), float(w), float(h)],
                    'class': class_name
                })
        
        return detections 