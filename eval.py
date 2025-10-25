import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from pathlib import Path
from dataloader import get_dataloaders
from lstm_model import LSTMModel
import cv2
import seaborn as sns
from scipy.stats import gaussian_kde
def visualize_attention(attention_weights, sequence_length, save_path=None):
    """Visualize attention weights as a heatmap"""
    plt.figure(figsize=(10, 5))
    if len(attention_weights.shape) == 1:
        attention_weights = attention_weights.reshape(1, -1)
    plt.imshow(attention_weights, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.xlabel('Sequence Position')
    plt.ylabel('Attention Head')
    plt.title('Attention Weights Heatmap')
    plt.xticks(range(sequence_length), [f't-{i}' for i in reversed(range(sequence_length))])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_on_images(images, attention_weights, save_path=None):
    """Overlay attention weights on input images"""
    plt.figure(figsize=(15, 5))
    num_images = min(5, len(images))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        attn = attention_weights[i] if len(attention_weights.shape) > 1 else attention_weights
        heatmap = cv2.resize(attn, (img.shape[1], img.shape[0]))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f'Frame {i}')
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_model(model, test_loader, device, sequence_length=5, save_dir="results"):
    """Main evaluation function with both visualization types"""
    model.eval()
    all_preds = []
    all_targets = []
    attention_weights = []
    sample_images = None
    with torch.no_grad():
        for batch_idx, ((picam_seq, webcam_seq, ball_seq, motor_seq), targets) in enumerate(tqdm(test_loader)):
            picam_seq = picam_seq.to(device)
            webcam_seq = webcam_seq.to(device)
            ball_seq = ball_seq.float().to(device)
            motor_seq = motor_seq.float().to(device)
            targets = targets.float().to(device)

            outputs = model(picam_seq=picam_seq, webcam_seq=webcam_seq, ball_coords=ball_seq, motor_seq=motor_seq)
            if hasattr(model, 'cross_attention') and hasattr(model.cross_attention, 'last_attn_weights'):
                attn = model.cross_attention.last_attn_weights.detach().cpu().numpy()
                attention_weights.append(attn)
                if batch_idx == 0:
                    sample_images = webcam_seq[0]
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # === DENORMALIZE to degrees ===
    preds = preds * 180.0
    targets = targets * 180.0

    os.makedirs(save_dir, exist_ok=True)
    if attention_weights and sample_images is not None:
        avg_attention = np.mean(np.concatenate(attention_weights), axis=0)
        visualize_attention(avg_attention, sequence_length, os.path.join(save_dir, "attention_heatmap.png"))
        visualize_attention_on_images(sample_images, avg_attention, os.path.join(save_dir, "attention_on_images.png"))
    else:
        avg_attention = None
    return preds, targets, avg_attention


def calculate_metrics(preds, targets):
    """Calculate regression metrics (now in degrees)"""
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    mse_per_dim, rmse_per_dim, mae_per_dim, r2_per_dim, std_per_dim, max_error_per_dim = [], [], [], [], [], []
    for i in range(preds.shape[1]):
        t, p = targets[:, i], preds[:, i]
        mse_i = mean_squared_error(t, p)
        rmse_i = np.sqrt(mse_i)
        mae_i = mean_absolute_error(t, p)
        r2_i = r2_score(t, p)
        std_i = np.std(t - p)
        max_i = np.max(np.abs(t - p))
        mse_per_dim.append(mse_i)
        rmse_per_dim.append(rmse_i)
        mae_per_dim.append(mae_i)
        r2_per_dim.append(r2_i)
        std_per_dim.append(std_i)
        max_error_per_dim.append(max_i)

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'mse_per_dim': mse_per_dim, 'rmse_per_dim': rmse_per_dim,
        'mae_per_dim': mae_per_dim, 'r2_per_dim': r2_per_dim,
        'std_per_dim': std_per_dim, 'max_error_per_dim': max_error_per_dim
    }


def plot_detailed_results(preds, targets, save_dir="results"):
    """Plots with angles in degrees"""
    sns.set_style("whitegrid")
    os.makedirs(save_dir, exist_ok=True)

    angle_metrics = {f"Angle {i+1}": {"r2": r2_score(targets[:, i], preds[:, i])} for i in range(targets.shape[1])}
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(18, 10))

    for i in range(3):
        plt.subplot(3, 2, 2 * i + 1)
        plt.plot(targets[:, i], label=f'True Angle {i+1}', alpha=0.7)
        plt.plot(preds[:, i], '--', label=f'Predicted Angle {i+1}', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Angle (degrees)')
        plt.title(f'Angle {i+1} Predictions Over Time')
        plt.legend()
        plt.grid(True)

    for i in range(3):
        plt.subplot(3, 2, 2 * i + 2)
        x, y = targets[:, i], preds[:, i]
        plt.scatter(x, y, alpha=0.5, label='Predictions')
        coeffs = np.polyfit(x, y, 1)
        plt.plot(x, np.poly1d(coeffs)(x), color='red', label='Regression Line')
        plt.plot([0, 180], [0, 180], 'g--', label='Perfect Prediction')
        plt.xlabel(f'True Angle {i+1} (°)')
        plt.ylabel(f'Predicted Angle {i+1} (°)')
        plt.title(f'Angle {i+1}: R²={angle_metrics[f"Angle {i+1}"]["r2"]:.3f}')
        plt.legend()

    plt.suptitle('Model Evaluation (in Degrees)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    # === Error Distribution ===
    # fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # for i, ax in enumerate(axs):
    #     errors = preds[:, i] - targets[:, i]
    #     ax.hist(errors, bins=30, density=True, alpha=0.6, label='Error Histogram')
    #     kde = gaussian_kde(errors)
    #     x_vals = np.linspace(min(errors), max(errors), 200)
    #     ax.plot(x_vals, kde(x_vals), color='blue', label='KDE')
    #     ax.set_title(f'Angle {i+1} Error Distribution (°)')
    #     ax.set_xlabel('Error (°)')
    #     ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, ax in enumerate(axs):
        errors = preds[:, i] - targets[:, i]
        ax.hist(errors, bins=30, density=True, alpha=0.6, label='Error Histogram')

        kde = gaussian_kde(errors)
        x_vals = np.linspace(errors.min(), errors.max(), 200)
        ax.plot(x_vals, kde(x_vals), color='blue', label='KDE')

            # Optional: mean & std lines for clarity
        mean_err = errors.mean()
        std_err = errors.std()
        x_vals = np.linspace(min(errors), max(errors), 200)
        ax.plot(x_vals, kde(x_vals), color='blue', label='Error Histogram (KDE)')
        ax.set_title(f'Angle {i+1} Error Distribution (°)')
        ax.set_xlabel('Error (°)')
        ax.legend()

        ax.axvline(mean_err, color='red', linestyle='--', label=f'Mean={mean_err:.2f}')
            
            # === Error Distribution ===

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')

    plt.show()
    print(f"[✅] Generated error distribution plots.")    
    

def save_results(preds, targets, metrics, attention_weights=None, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "predictions.npz"), predictions=preds, targets=targets)
    if attention_weights is not None:
        np.save(os.path.join(save_dir, "attention_weights.npy"), attention_weights)

    with open(os.path.join(save_dir, "metrics.txt"), 'w') as f:
        f.write("=== Overall Metrics (Degrees) ===\n")
        f.write(f"MSE: {metrics['mse']:.4f}\nRMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}\n\n")
        for i in range(len(metrics['mae_per_dim'])):
            f.write(f"Joint {i+1}: MAE={metrics['mae_per_dim'][i]:.4f}, RMSE={metrics['rmse_per_dim'][i]:.4f}, "
                    f"R²={metrics['r2_per_dim'][i]:.4f}, STD={metrics['std_per_dim'][i]:.4f}, "
                    f"MaxErr={metrics['max_error_per_dim'][i]:.4f}\n")
    print(f"Results saved to {save_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_arm_control_lstm_working.pth"
    sequence_length = 5

    model = LSTMModel(motor_dim=3, hidden_dim=512, output_dim=3, num_layers=2, dropout=0.3, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"Model loaded on {device}")

    base_path = "E:/Bibek_Files/robo_yolo"
    episodes = [f"episode_{i}" for i in range(1, 11)]
    pi_cam_dirs = [f"{base_path}/dataset/{ep}/picam_frames" for ep in episodes]
    web_cam_dirs = [f"{base_path}/dataset/{ep}/webcam_frames" for ep in episodes]
    csv_paths = [f"{base_path}/dataset/{ep}/data.csv" for ep in episodes]
    yolo_dirs = [f"{base_path}/yolo_outputs/{ep}/webcam_frames/predicted_labels" for ep in episodes]

    _, _,test_loader = get_dataloaders(
        pi_cam_dirs=pi_cam_dirs, web_cam_dirs=web_cam_dirs,
        csv_paths=csv_paths, yolo_dirs=yolo_dirs,
        batch_size=32, sequence_length=sequence_length, img_size=(224, 224)
    )

    preds, targets, attention_weights = evaluate_model(model, test_loader, device, sequence_length)
    metrics = calculate_metrics(preds, targets)

    print("\nOverall Metrics (Degrees):")
    print(f"MAE={metrics['mae']:.2f}°, RMSE={metrics['rmse']:.2f}°, R²={metrics['r2']:.3f}")
    print("Per-Joint Metrics:")
    for i in range(len(metrics['mae_per_dim'])):
        print(f"  Joint {i+1}: MAE={metrics['mae_per_dim'][i]:.2f}°, RMSE={metrics['rmse_per_dim'][i]:.2f}°, "
              f"R²={metrics['r2_per_dim'][i]:.3f}")

    save_dir = "results"
    plot_detailed_results(preds, targets, save_dir=save_dir)
    save_results(preds, targets, metrics, attention_weights, save_dir)


if __name__ == "__main__":
    main()
