import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple
import random
from collections import defaultdict

class ArmSequenceDataset(Dataset):
    def __init__(self, 
                 pi_cam_dirs: List[str],
                 web_cam_dirs: List[str],
                 csv_paths: List[str],
                 yolo_dirs: List[str],
                 sequence_length: int = 5, 
                 img_size: Tuple[int,int] = (224, 224)):
        self.sequences = []
        self.targets = []
        self.sequence_length = sequence_length
        self.img_size = img_size

        stats = {
            'total_frames': 0,
            'loaded_frames': 0,
            'missing_images': 0,
            'missing_csv': 0,
            'missing_yolo': 0
        }

        for pi_dir, web_dir, csv_path, yolo_dir in zip(pi_cam_dirs, web_cam_dirs, csv_paths, yolo_dirs):
            pi_cam_dir = Path(pi_dir)
            web_cam_dir = Path(web_dir)
            yolo_pred_dir = Path(yolo_dir)
            csv_path = Path(csv_path)

            if not (pi_cam_dir.exists() and web_cam_dir.exists()):
                stats['missing_images'] += 1
                continue
            if not csv_path.exists():
                stats['missing_csv'] += 1
                continue
            if not yolo_pred_dir.exists():
                stats['missing_yolo'] += 1
                continue

            df = pd.read_csv(csv_path)
            timestamp_to_idx = {
                row['timestamp'].replace(':', '_'): idx
                for idx, row in df.iterrows()
            }

            valid_frames = []

            for img_file in sorted(pi_cam_dir.glob("*.jpg")):
                timestamp = img_file.stem
                webcam_file = web_cam_dir / f"{timestamp}.jpg"
                yolo_txt = yolo_pred_dir / f"{timestamp}.txt"

                stats['total_frames'] += 1

                if not webcam_file.exists() or not yolo_txt.exists() or timestamp not in timestamp_to_idx:
                    continue

                try:
                    # Read YOLO .txt: assume only 1 object per frame
                    with open(yolo_txt, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            continue  # skip empty files
                        class_id, x, y, w, h = map(float, lines[0].split())
                        yolo_coords = np.array([x, y, w, h], dtype=np.float32)

                    # Load and resize images
                    picam_img = cv2.imread(str(img_file))
                    webcam_img = cv2.imread(str(webcam_file))

                    picam_img = cv2.cvtColor(picam_img, cv2.COLOR_BGR2RGB)
                    webcam_img = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2RGB)


                    picam_img = cv2.resize(picam_img, self.img_size)
                    webcam_img = cv2.resize(webcam_img, self.img_size)

                    picam_img = torch.from_numpy(picam_img).permute(2,0,1).float() / 255.0
                    webcam_img = torch.from_numpy(webcam_img).permute(2,0,1).float() / 255.0

                    angles = df.iloc[timestamp_to_idx[timestamp]][['angle0', 'angle1', 'angle2']].values.astype(np.float32)
                    angles[0:2] = angles[0:2] / 180.0 
                    angles[2] = angles[2]/180.0
                    motor_angles = angles

                    
                    valid_frames.append({
                        'picam': picam_img,
                        'webcam': webcam_img,
                        'ball': yolo_coords,
                        'motors': motor_angles
                    })

                    stats['loaded_frames'] += 1

                except Exception as e:
                    print(f"Error loading frame {timestamp}: {e}")
                    continue

            # Create sequences
            if len(valid_frames) >= sequence_length:
                for i in range(len(valid_frames) - sequence_length ):
                    sequence = {
                        'picam_seq': torch.stack([f['picam'] for f in valid_frames[i:i+sequence_length]]),
                        'webcam_seq': torch.stack([f['webcam'] for f in valid_frames[i:i+sequence_length]]),
                        'ball_seq': np.array([f['ball'] for f in valid_frames[i:i+sequence_length]]),
                        'motor_seq': np.array([f['motors'] for f in valid_frames[i:i+sequence_length]])
                    }
                    target = valid_frames[i+sequence_length]['motors']
                    self.sequences.append(sequence)
                    self.targets.append(target)

        print(f"\nDataset statistics:")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Successfully loaded frames: {stats['loaded_frames']}")
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        print(f"Issues: Missing images={stats['missing_images']}, Missing CSVs={stats['missing_csv']}, Missing YOLO={stats['missing_yolo']}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            seq['picam_seq'],
            seq['webcam_seq'],
            seq['ball_seq'],
            seq['motor_seq']
        ), self.targets[idx]


def get_dataloaders(
    pi_cam_dirs: List[str],
    web_cam_dirs: List[str],
    csv_paths: List[str],
    yolo_dirs: List[str],
    batch_size: int = 32,
    sequence_length: int = 5,
    img_size: Tuple[int,int] = (224, 224),
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    combined = list(zip(pi_cam_dirs, web_cam_dirs, csv_paths, yolo_dirs))
    random.seed(random_seed)
    random.shuffle(combined)
    pi_cam_dirs, web_cam_dirs, csv_paths, yolo_dirs = zip(*combined)

    num_total = len(pi_cam_dirs)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    train_dataset = ArmSequenceDataset(
        pi_cam_dirs=pi_cam_dirs[:num_train],
        web_cam_dirs=web_cam_dirs[:num_train],
        csv_paths=csv_paths[:num_train],
        yolo_dirs=yolo_dirs[:num_train],
        sequence_length=sequence_length,
        img_size=img_size
    )
    val_dataset = ArmSequenceDataset(
        pi_cam_dirs=pi_cam_dirs[num_train:num_train+num_val],
        web_cam_dirs=web_cam_dirs[num_train:num_train+num_val],
        csv_paths=csv_paths[num_train:num_train+num_val],
        yolo_dirs=yolo_dirs[num_train:num_train+num_val],
        sequence_length=sequence_length,
        img_size=img_size
    )
    test_dataset = ArmSequenceDataset(
        pi_cam_dirs=pi_cam_dirs[num_train+num_val:],
        web_cam_dirs=web_cam_dirs[num_train+num_val:],
        csv_paths=csv_paths[num_train+num_val:],
        yolo_dirs=yolo_dirs[num_train+num_val:],
        sequence_length=sequence_length,
        img_size=img_size
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


if __name__ == "__main__":
    base_path = "E:/Bibek_Files/robo_yolo"
    episodes = [f"episode_{i}" for i in range(1, 11)]

    pi_cam_dirs = [f"{base_path}/dataset/{ep}/picam_frames" for ep in episodes]
    web_cam_dirs = [f"{base_path}/dataset/{ep}/webcam_frames" for ep in episodes]
    csv_paths = [f"{base_path}/dataset/{ep}/data.csv" for ep in episodes]
    yolo_dirs = [f"{base_path}/yolo_outputs/{ep}/webcam_frames/predicted_labels" for ep in episodes]

    train_loader, val_loader, test_loader = get_dataloaders(
        pi_cam_dirs=pi_cam_dirs,
        web_cam_dirs=web_cam_dirs,
        csv_paths=csv_paths,
        yolo_dirs=yolo_dirs,
        batch_size=16,
        sequence_length=5,
        img_size=(224, 224)
    )

    print(" Dataloaders created successfully!")
