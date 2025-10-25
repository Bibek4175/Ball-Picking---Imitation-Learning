import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os
from collections import deque
from tqdm import tqdm
from lstm_model import LSTMModel 

import torch.nn as nn
from ultralytics import YOLO

class BallTrackingPredictor:
    def __init__(self, yolo_weights_path, lstm_weights_path, buffer_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def detect_ball(self, img_np, conf_threshold=0.2):
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

    def process_frame_pair(self, pi_frame, web_frame, prev_angles):
        """Process a pair of frames from both cameras"""
        # Convert to tensors
        pi_tensor = self.transform(Image.fromarray(pi_frame)).to(self.device)
        web_tensor = self.transform(Image.fromarray(web_frame)).to(self.device)

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
            prev_angles["angle3"] / 90.0
        ], device=self.device)
        self.motor_buffer.append(norm_angles)

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

            # Update motor buffer with new prediction
            predicted_angles = (output.squeeze(0).cpu().numpy())

            angle1 = predicted_angles[0] * 180.0
            angle2 = predicted_angles[1] * 180.0
            angle3 = predicted_angles[2] * 90.0
            return {
                "angle1": float(angle1),
                "angle2": float(angle2),
                "angle3": float(angle3),
                "ball_detected": bbox is not None,
                "confidence": conf,
                "frame": pi_frame
            }
        return None

def evaluate_frame_folders(predictor, picam_folder, webcam_folder):
    """Process all frames in the folders"""
    picam_frames = sorted([f for f in os.listdir(picam_folder) if f.endswith('.jpg')])
    webcam_frames = sorted([f for f in os.listdir(webcam_folder) if f.endswith('.jpg')])

    assert len(picam_frames) == len(webcam_frames), "Frame count mismatch"

    all_preds = []
    prev_angles = {"angle1": 90, "angle2": 90, "angle3": 90}  # Initialize to 90 degrees

    for picam_file, webcam_file in tqdm(zip(picam_frames, webcam_frames),
                                      total=len(picam_frames),
                                      desc="Processing frames"):
        # Load images
        pi_frame = cv2.imread(os.path.join(picam_folder, picam_file))
        pi_frame = cv2.cvtColor(pi_frame, cv2.COLOR_BGR2RGB)
        web_frame = cv2.imread(os.path.join(webcam_folder, webcam_file))
        web_frame = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)

        # Process frame pair
        result = predictor.process_frame_pair(pi_frame, web_frame, prev_angles)

        if result:
            all_preds.append([result["angle1"], result["angle2"], result["angle3"]])
            # Update previous angles for next frame
            prev_angles = {
                "angle1": result["angle1"],
                "angle2": result["angle2"],
                "angle3": result["angle3"]
            }

    return np.array(all_preds)

def visualize_frame_predictions(picam_folder, preds, output_folder="output_frames"):
    """Overlay predictions on frames and save results"""
    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(picam_folder) if f.endswith('.jpg')])

    # Adjust for buffer delay
    start_idx = predictor.buffer_size - 1
    pred_idx = 0

    for i, frame_file in enumerate(tqdm(frame_files, desc="Generating output")):
        frame = cv2.imread(os.path.join(picam_folder, frame_file))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Only show predictions after buffer fills
        if i >= start_idx and pred_idx < len(preds):
            angles = preds[pred_idx]

            # Draw predictions
            cv2.putText(frame, f"Predicted Angles:", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"1: {angles[0]:.1f}°", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"2: {angles[1]:.1f}°", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"3: {angles[2]:.1f}°", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            pred_idx += 1

        cv2.imwrite(os.path.join(output_folder, frame_file), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Saved annotated frames to {output_folder}")

if __name__ == "__main__":
    # Initialize predictor
    predictor = BallTrackingPredictor(
        yolo_weights_path="E:/Bibek_Files/robo_yolo/yolov8_model/content/runs/detect/train3/weights/last.pt",
        lstm_weights_path="E:/Bibek_Files/robo_yolo/best_arm_control_lstm.pth",
        buffer_size=5
    )

    # Frame folders
    picam_folder = "E:/Bibek_Files/robo_yolo/yolov8_model/episode-20250723-112958/picam_frames"
    webcam_folder = "E:/Bibek_Files/robo_yolo/yolov8_model/episode-20250723-112958/webcam_frames"

    # Run evaluation
    preds = evaluate_frame_folders(predictor, picam_folder, webcam_folder)

    # Visualize results
    visualize_frame_predictions(webcam_folder, preds)

    # Save predictions
    np.save("frame_predictions.npy", preds)
    print(f"Average predicted angles: {np.mean(preds, axis=0)}")