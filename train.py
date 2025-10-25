import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import numpy as np
from lstm_model import LSTMModel  # Make sure this is your updated model class
from dataloader import get_dataloaders  # Updated dataloader from previous step

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5       
              )

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/{config.experiment_name}_{timestamp}')

        # Training statistics
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, ((picam_seq, webcam_seq, ball_seq, motor_seq), targets) in enumerate(progress_bar):
            # Move data to device
            picam_seq = picam_seq.to(self.device)  # [batch, seq_len, 3, H, W]
            webcam_seq = webcam_seq.to(self.device)
            ball_seq = ball_seq.float().to(self.device)  # [batch, seq_len, 4]
            motor_seq = motor_seq.float().to(self.device)  # [batch, seq_len, 3]
            targets = targets.float().to(self.device)  # [batch, 3]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                picam_seq=picam_seq,
                webcam_seq=webcam_seq,
                ball_coords=ball_seq,
                motor_seq=motor_seq
            )

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * picam_seq.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

            # Log batch loss to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        return total_loss / len(self.train_loader.dataset)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}")

        with torch.no_grad():
            for batch_idx, ((picam_seq, webcam_seq, ball_seq, motor_seq), targets) in enumerate(progress_bar):
                # Move data to device
                picam_seq = picam_seq.to(self.device)
                webcam_seq = webcam_seq.to(self.device)
                ball_seq = ball_seq.float().to(self.device)
                motor_seq = motor_seq.float().to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = self.model(
                    picam_seq=picam_seq,
                    webcam_seq=webcam_seq,
                    ball_coords=ball_seq,
                    motor_seq=motor_seq
                )

                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.detach().item() * picam_seq.size(0)
                progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_loss / len(self.val_loader.dataset)
        return avg_val_loss

    def train(self):
        for epoch in range(self.config.epochs):
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch statistics
            print(f"\nEpoch {epoch + 1:03d}/{self.config.epochs:03d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), f"best_{self.config.experiment_name}.pth")
                print(f"Saved new best model with val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        self.writer.close()
        print("\nTraining completed!")

def train_in_colab(config):
    """Main training function with configuration dictionary"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_path = "E:/Bibek_Files/robo_yolo"
    episodes = [f"episode_{i}" for i in range(1, 11)]  # Example episode names
    
    pi_cam_dirs = [f"{base_path}/dataset/{ep}/picam_frames" for ep in episodes]
    web_cam_dirs = [f"{base_path}/dataset/{ep}/webcam_frames" for ep in episodes]
    csv_paths = [f"{base_path}/dataset/{ep}/data.csv" for ep in episodes]
    yolo_dirs = [f"{base_path}/yolo_outputs/{ep}/webcam_frames/predicted_labels" for ep in episodes]
    
    train_loader, val_loader, test_loader = get_dataloaders(
        pi_cam_dirs=pi_cam_dirs,
        web_cam_dirs=web_cam_dirs,
        csv_paths=csv_paths,
        yolo_dirs=yolo_dirs,
        batch_size=32,
        sequence_length=5,
        img_size=(224, 224)
    )
    print("dataloaders created successfully!")

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['sequence_length']}")

    # Initialize model
    print("\nInitializing model...")
    model = LSTMModel(
        motor_dim=3,
        hidden_dim=config['hidden_dim'],
        output_dim=3,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device
    )

    # Print model summary
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Convert config dict to namespace for Trainer
    class ConfigNamespace:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    trainer_config = ConfigNamespace(config)

    # Train the model
    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config
    )
    trainer.train()

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'batch_size': 32,
        'sequence_length': 5,
        'img_size': 224,
        'lr': 1e-3,
        'epochs': 50,
        'patience': 10,
        'weight_decay': 1e-4,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.2,
        'experiment_name': "arm_control_lstm"
    }

    # Print configuration
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"{key:>20}: {value}")

    # Start training
    train_in_colab(config)