import torch
from fastapi import FastAPI, WebSocket
from collections import deque
import numpy as np
from io import BytesIO
from torchvision import transforms
from PIL import Image
import json
import uvicorn
import cv2
import os
from datetime import datetime
from lstm_model import LSTMModel
from ultralytics import YOLO

app = FastAPI()

# Configuration
BUFFER_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "detection_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BallTrackingPredictor:
    def __init__(self, yolo_weights_path, lstm_weights_path, buffer_size=5):
        self.device = DEVICE
        self.buffer_size = buffer_size

        # Initialize models
        self.load_models(yolo_weights_path, lstm_weights_path)
        self.initialize_buffers()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_models(self, yolo_path, lstm_path):
        """Load all required models"""
        # YOLO for ball detection
        self.yolo = YOLO(yolo_path).to(self.device)

        # LSTM model
        self.lstm = LSTMModel(
            hidden_dim=512,
            output_dim=3,
            motor_dim=3,
            num_layers=2,
            dropout=0.2,
            device=self.device
        ).to(self.device)
        self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
        self.lstm.eval()

    def initialize_buffers(self):
        """Initialize buffers for temporal processing"""
        self.picam_buffer = deque(maxlen=self.buffer_size)
        self.webcam_buffer = deque(maxlen=self.buffer_size)
        self.ball_buffer = deque(maxlen=self.buffer_size)
        self.motor_buffer = deque(maxlen=self.buffer_size)

    def detect_ball(self, img_np, conf_threshold=0.4):
        """Detect ball using YOLO and return coordinates"""
        with torch.no_grad():
            results = self.yolo(img_np, conf=conf_threshold)
            boxes = results[0].boxes

            if len(boxes) == 0:
                return None, 0.0

            best_idx = boxes.conf.argmax()
            bbox = boxes.xywhn[best_idx].cpu().numpy()
            conf = boxes.conf[best_idx].item()
            return torch.tensor(bbox, dtype=torch.float32).to(self.device), conf

    def save_debug_image(self, img_np, bbox, conf, angles, timestamp):
        debug_img = img_np.copy()
        h, w = debug_img.shape[:2]
        
        if bbox is not None:
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        info = f"Conf: {conf:.2f}" if conf else "No detection"
        info += f" | Angles: {angles}"
        cv2.putText(debug_img, info, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        filename = f"{OUTPUT_DIR}/det_{timestamp}.jpg"
        cv2.imwrite(filename, debug_img)

    def process_frame_pair(self, pi_frame_bytes, web_frame_bytes, prev_angles):
        """Process a pair of frames from both cameras"""
        # Convert bytes to numpy arrays
        pi_img = Image.open(BytesIO(pi_frame_bytes)).convert("RGB")
        web_img = Image.open(BytesIO(web_frame_bytes)).convert("RGB")
        
        pi_frame = np.array(pi_img)
        web_frame = np.array(web_img)
        
        # Convert to tensors
        try:
            pi_tensor = self.transform(pi_img).to(self.device)
            web_tensor = self.transform(web_img).to(self.device)
        except:
            # Handle cases where transform fails (e.g., empty image)
            return None

        # Update image buffers
        self.picam_buffer.append(pi_tensor)
        self.webcam_buffer.append(web_tensor)

        # Detect ball and update buffer
        bbox, conf = self.detect_ball(web_frame)
        self.ball_buffer.append(bbox if bbox is not None else torch.zeros(4).to(self.device))

        # Normalize angles (scale to [0, 1])
        norm_angles = torch.tensor([
            prev_angles["angle1"] / 180.0,
            prev_angles["angle2"] / 180.0,
            prev_angles["angle3"] / 180.0
        ], device=self.device)
        self.motor_buffer.append(norm_angles)

        # Create debug info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        angle_str = f"{prev_angles['angle1']:.1f},{prev_angles['angle2']:.1f},{prev_angles['angle3']:.1f}"
        self.save_debug_image(web_frame, bbox, conf, angle_str, timestamp)

        # Only predict when buffer is full
        if len(self.motor_buffer) == self.buffer_size:
            # Prepare input tensors
            picam_seq = torch.stack(list(self.picam_buffer)).unsqueeze(0)  # [1, seq_len, 3, H, W]
            webcam_seq = torch.stack(list(self.webcam_buffer)).unsqueeze(0)  # [1, seq_len, 3, H, W]
            ball_seq = torch.stack(list(self.ball_buffer)).unsqueeze(0)  # [1, seq_len, 4]
            motor_seq = torch.stack(list(self.motor_buffer)).unsqueeze(0)  # [1, seq_len, 3]

            with torch.no_grad():
                output = self.lstm(
                    picam_seq=picam_seq,
                    webcam_seq=webcam_seq,
                    ball_coords=ball_seq,
                    motor_seq=motor_seq
                )

            # Convert output to angles (0-180)
            predicted_angles = (output.squeeze(0).cpu().numpy()) 
            predicted_angles[0] = predicted_angles[0] * 180.0
            predicted_angles[1] = predicted_angles[1] * 180.0
            predicted_angles[2] = predicted_angles[2] * 180.0
            return {
                "angle1": float(predicted_angles[0]),
                "angle2": float(predicted_angles[1]),
                "angle3": float(predicted_angles[2]),
                "ball_detected": bbox is not None,
                "confidence": conf
            }
        return None

# Initialize predictor
print("Loading models...")
predictor = BallTrackingPredictor(
    yolo_weights_path="E:/Bibek_Files/robo_yolo/yolov8_model/content/runs/detect/train3/weights/best.pt",
    lstm_weights_path="E:/Bibek_Files/robo_yolo/best_arm_control_lstm_working.pth",
    buffer_size=BUFFER_SIZE
)


@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive frames - webcam first since it's now our detection source
            pi_data = await websocket.receive_bytes()
            web_data = await websocket.receive_bytes()
            angle_json = await websocket.receive_text()
            prev_angles = json.loads(angle_json)
            
            # Process frame pair
            result = predictor.process_frame_pair(pi_data, web_data, prev_angles)
            
            if result:
                response = {
                    "angle1": result["angle1"],
                    "angle2": result["angle2"],
                    "angle3": result["angle3"]
                }
                await websocket.send_json(response)
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)