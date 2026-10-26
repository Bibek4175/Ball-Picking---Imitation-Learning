import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import json
from lstm_model import LSTMModel
from dataloader import get_dataloaders

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 5
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

DATA_PATH = "E:/Bibek_Files/robo_yolo"
EPISODES = [f"episode_{i}" for i in range(1, 11)]

# Checkpoint paths
CKPT_NOCA = "E:/Bibek_Files/imitation_research/best_arm_control_resnet_test.pth"
CKPT_CA   = "E:/Bibek_Files/imitation_research/best_arm_control_lstm_test.pth"

# Joint names for better visualization
JOINT_NAMES = ['Joint 1', 'Joint 2', 'Joint 3']

def calculate_comprehensive_metrics(preds, targets, joint_names):
    """Calculate comprehensive metrics including per-joint statistics"""
    metrics = {}
    
    # Overall metrics
    metrics['overall'] = {
        'mse': mean_squared_error(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae': mean_absolute_error(targets, preds),
        'r2': r2_score(targets, preds),
        'explained_variance': 1 - np.var(targets - preds) / np.var(targets)
    }
    
    # Per-joint metrics
    metrics['per_joint'] = {}
    for i, joint_name in enumerate(joint_names):
        joint_preds = preds[:, i]
        joint_targets = targets[:, i]
        errors = joint_preds - joint_targets
        
        metrics['per_joint'][joint_name] = {
            'mse': mean_squared_error(joint_targets, joint_preds),
            'rmse': np.sqrt(mean_squared_error(joint_targets, joint_preds)),
            'mae': mean_absolute_error(joint_targets, joint_preds),
            'r2': r2_score(joint_targets, joint_preds),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'error_95_percentile': np.percentile(np.abs(errors), 95),
            'correlation': np.corrcoef(joint_targets, joint_preds)[0, 1]
        }
    
    # Statistical analysis
    metrics['statistics'] = {
        'prediction_range': {
            'min': preds.min(),
            'max': preds.max(),
            'mean': preds.mean(),
            'std': preds.std()
        },
        'target_range': {
            'min': targets.min(),
            'max': targets.max(),
            'mean': targets.mean(),
            'std': targets.std()
        },
        'total_samples': len(preds)
    }
    
    return metrics
def print_metrics_comparison(metrics_noCA, metrics_CA, joint_names):
    """Print comprehensive comparison of metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # ------------------------------------------------
    # 1. Overall metrics comparison
    # ------------------------------------------------
    print("\n📊 OVERALL PERFORMANCE METRICS:")
    print("-" * 70)
    print(f"{'Metric':<15} {'No Cross-Attention':<20} {'With Cross-Attention':<20} {'Δ (CA - NoCA)':<15}")
    print("-" * 70)
    
    overall_metrics = ['mse', 'rmse', 'mae', 'r2']
    for metric in overall_metrics:
        val_noCA = metrics_noCA['overall'][metric]
        val_CA = metrics_CA['overall'][metric]
        improvement = val_CA - val_noCA
        print(f"{metric.upper():<15} {val_noCA:<20.4f} {val_CA:<20.4f} {improvement:+.4f}")
    
    # ------------------------------------------------
    # 2. Per-joint detailed comparison (MSE, RMSE, MAE, R²)
    # ------------------------------------------------
    print(f"\n🔩 PER-JOINT PERFORMANCE DETAILS:")
    print("-" * 90)
    print(f"{'Joint':<12} {'Metric':<8} {'No Cross-Attention':<20} {'With Cross-Attention':<20} {'Δ (CA - NoCA)':<15}")
    print("-" * 90)
    
    per_joint_metrics = ['mse', 'rmse', 'mae', 'r2']
    for joint_name in joint_names:
        for metric in per_joint_metrics:
            val_noCA = metrics_noCA['per_joint'][joint_name][metric]
            val_CA = metrics_CA['per_joint'][joint_name][metric]
            improvement = val_CA - val_noCA
            print(f"{joint_name:<12} {metric.upper():<8} {val_noCA:<20.4f} {val_CA:<20.4f} {improvement:+.4f}")
        print("-" * 90)
    
    # ------------------------------------------------
    # 3. Best performing joints (lowest MAE)
    # ------------------------------------------------
    print(f"\n🏆 BEST PERFORMING JOINTS (Lowest MAE):")
    print("-" * 40)
    joint_mae_noCA = {name: metrics_noCA['per_joint'][name]['mae'] for name in joint_names}
    joint_mae_CA = {name: metrics_CA['per_joint'][name]['mae'] for name in joint_names}
    
    best_joint_noCA = min(joint_mae_noCA, key=joint_mae_noCA.get)
    best_joint_CA = min(joint_mae_CA, key=joint_mae_CA.get)
    
    print(f"No Cross-Attention: {best_joint_noCA} (MAE: {joint_mae_noCA[best_joint_noCA]:.4f})")
    print(f"With Cross-Attention: {best_joint_CA} (MAE: {joint_mae_CA[best_joint_CA]:.4f})")


def create_comprehensive_plots(metrics_noCA, metrics_CA, joint_names, save_dir):
    """Create comprehensive comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Overall metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    overall_metrics = ['mse', 'rmse', 'mae', 'r2']
    metric_titles = ['MSE', 'RMSE', 'MAE', 'R² Score']
    
    for idx, (metric, title) in enumerate(zip(overall_metrics, metric_titles)):
        values = [metrics_noCA['overall'][metric], metrics_CA['overall'][metric]]
        colors = ['skyblue', 'orange']
        
        bars = axes[idx].bar(['No Cross-Attention', 'With Cross-Attention'], values, color=colors, alpha=0.7)
        axes[idx].set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(title)
        
        # Add value annotations
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                          f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_metrics_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Per-joint MAE comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(joint_names))
    width = 0.35
    
    mae_noCA = [metrics_noCA['per_joint'][name]['mae'] for name in joint_names]
    mae_CA = [metrics_CA['per_joint'][name]['mae'] for name in joint_names]
    
    plt.bar(x - width/2, mae_noCA, width, label='No Cross-Attention', color='skyblue', alpha=0.7)
    plt.bar(x + width/2, mae_CA, width, label='With Cross-Attention', color='orange', alpha=0.7)
    
    plt.xlabel('Joints')
    plt.ylabel('MAE')
    plt.title('Per-Joint MAE Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, joint_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (v1, v2) in enumerate(zip(mae_noCA, mae_CA)):
        plt.text(i - width/2, v1 + 0.01 * max(mae_noCA + mae_CA), f'{v1:.3f}', 
                ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, v2 + 0.01 * max(mae_noCA + mae_CA), f'{v2:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_joint_mae_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Per-joint RMSE comparison
    plt.figure(figsize=(12, 8))
    
    rmse_noCA = [metrics_noCA['per_joint'][name]['rmse'] for name in joint_names]
    rmse_CA = [metrics_CA['per_joint'][name]['rmse'] for name in joint_names]
    
    plt.bar(x - width/2, rmse_noCA, width, label='No Cross-Attention', color='lightcoral', alpha=0.7)
    plt.bar(x + width/2, rmse_CA, width, label='With Cross-Attention', color='lightgreen', alpha=0.7)
    
    plt.xlabel('Joints')
    plt.ylabel('RMSE')
    plt.title('Per-Joint RMSE Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, joint_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (v1, v2) in enumerate(zip(rmse_noCA, rmse_CA)):
        plt.text(i - width/2, v1 + 0.01 * max(rmse_noCA + rmse_CA), f'{v1:.3f}', 
                ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, v2 + 0.01 * max(rmse_noCA + rmse_CA), f'{v2:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_joint_rmse_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Error distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, joint_name in enumerate(joint_names):
        errors_noCA = metrics_noCA['per_joint'][joint_name]['std_error']
        errors_CA = metrics_CA['per_joint'][joint_name]['std_error']
        
        error_data = [errors_noCA, errors_CA]
        colors = ['skyblue', 'orange']
        
        bars = axes[i].bar(['No CA', 'With CA'], error_data, color=colors, alpha=0.7)
        axes[i].set_title(f'{joint_name}\nError Standard Deviation', fontweight='bold')
        axes[i].set_ylabel('Std. Deviation')
        
        for bar, value in zip(bars, error_data):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(error_data),
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_std_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

def save_comprehensive_results(metrics_noCA, metrics_CA, preds_noCA, preds_CA, targets, save_dir):
    """Save comprehensive results to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save numpy arrays
    np.savez(os.path.join(save_dir, "comparison_predictions.npz"),
             targets=targets,
             preds_noCA=preds_noCA,
             preds_CA=preds_CA)
    
    # Save metrics as JSON
    comparison_metrics = {
        'no_cross_attention': metrics_noCA,
        'with_cross_attention': metrics_CA
    }
    
    # Convert numpy values to Python floats for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(os.path.join(save_dir, "comparison_metrics.json"), 'w') as f:
        json.dump(convert_numpy(comparison_metrics), f, indent=2)
    
    # Save detailed text report
    with open(os.path.join(save_dir, "detailed_comparison_report.txt"), 'w') as f:
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<15} {'No CA':<12} {'With CA':<12} {'Difference':<12}\n")
        f.write("-" * 40 + "\n")
        
        overall_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in overall_metrics:
            noCA_val = metrics_noCA['overall'][metric]
            CA_val = metrics_CA['overall'][metric]
            diff = CA_val - noCA_val
            f.write(f"{metric.upper():<15} {noCA_val:<12.4f} {CA_val:<12.4f} {diff:+.4f}\n")
        
        f.write("\nPER-JOINT MAE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Joint':<15} {'No CA':<12} {'With CA':<12} {'Difference':<12}\n")
        f.write("-" * 40 + "\n")
        
        for joint_name in JOINT_NAMES:
            noCA_mae = metrics_noCA['per_joint'][joint_name]['mae']
            CA_mae = metrics_CA['per_joint'][joint_name]['mae']
            diff = CA_mae - noCA_mae
            f.write(f"{joint_name:<15} {noCA_mae:<12.4f} {CA_mae:<12.4f} {diff:+.4f}\n")
    
    print(f"📁 Results saved to: {save_dir}")

def main():
    # ----------------------------
    # Load Data
    # ----------------------------
    pi_cam_dirs = [f"{DATA_PATH}/dataset/{ep}/picam_frames" for ep in EPISODES]
    web_cam_dirs = [f"{DATA_PATH}/dataset/{ep}/webcam_frames" for ep in EPISODES]
    csv_paths    = [f"{DATA_PATH}/dataset/{ep}/data.csv" for ep in EPISODES]
    yolo_dirs    = [f"{DATA_PATH}/yolo_outputs/{ep}/webcam_frames/predicted_labels" for ep in EPISODES]

    train_loader, val_loader, test_loader = get_dataloaders(
        pi_cam_dirs, web_cam_dirs, csv_paths, yolo_dirs,
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        img_size=IMG_SIZE
    )
    print("✅ Test data loaded successfully")

    # ----------------------------
    # Initialize models
    # ----------------------------
    def load_model(use_cross_attention, ckpt_path):
        model = LSTMModel(
            hidden_dim=512,
            output_dim=3,
            motor_dim=3,
            num_layers=2,
            dropout=0.2,
            ball_scale=3.0,
            use_cross_attention=use_cross_attention,
            device=DEVICE
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model

    model_noCA = load_model(use_cross_attention=False, ckpt_path=CKPT_NOCA)
    model_CA   = load_model(use_cross_attention=True,  ckpt_path=CKPT_CA)

    # ----------------------------
    # Evaluation function
    # ----------------------------
    def evaluate_model(model, loader):
        all_preds, all_targets = [], []

        with torch.no_grad():
            for (picam_seq, webcam_seq, ball_seq, motor_seq), targets in tqdm(loader):
                picam_seq = picam_seq.to(DEVICE)
                webcam_seq = webcam_seq.to(DEVICE)
                ball_seq   = ball_seq.float().to(DEVICE)
                motor_seq  = motor_seq.float().to(DEVICE)
                targets    = targets.float().to(DEVICE)

                preds = model(picam_seq, webcam_seq, ball_seq, motor_seq)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds   = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        return all_preds, all_targets

    # ----------------------------
    # Evaluate both models
    # ----------------------------
    print("\nEvaluating model WITHOUT Cross-Attention...")
    preds_noCA, targets = evaluate_model(model_noCA, test_loader)

    print("\nEvaluating model WITH Cross-Attention...")
    preds_CA, _ = evaluate_model(model_CA, test_loader)

    # ----------------------------
    # Calculate comprehensive metrics
    # ----------------------------
    print("\n📊 Calculating comprehensive metrics...")
    metrics_noCA = calculate_comprehensive_metrics(preds_noCA, targets, JOINT_NAMES)
    metrics_CA = calculate_comprehensive_metrics(preds_CA, targets, JOINT_NAMES)

    # ----------------------------
    # Print comparison results
    # ----------------------------
    print_metrics_comparison(metrics_noCA, metrics_CA, JOINT_NAMES)

    # ----------------------------
    # Create comprehensive visualizations
    # ----------------------------
    print("\n🖼️  Generating comprehensive visualizations...")
    save_dir = "comparison_results"
    create_comprehensive_plots(metrics_noCA, metrics_CA, JOINT_NAMES, save_dir)

    # ----------------------------
    # Save comprehensive results
    # ----------------------------
    save_comprehensive_results(metrics_noCA, metrics_CA, preds_noCA, preds_CA, targets, save_dir)

    # ----------------------------
    # Final summary
    # ----------------------------
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY")
    print("="*80)
    
    overall_mae_noCA = metrics_noCA['overall']['mae']
    overall_mae_CA = metrics_CA['overall']['mae']
    improvement = overall_mae_CA - overall_mae_noCA
    
    if improvement < 0:
        print(f"✅ Cross-Attention IMPROVES performance by {abs(improvement):.4f} MAE")
    else:
        print(f"❌ Cross-Attention DEGRADES performance by {improvement:.4f} MAE")
    
    best_model = "WITH Cross-Attention" if overall_mae_CA < overall_mae_noCA else "WITHOUT Cross-Attention"
    print(f"🏆 Best performing model: {best_model}")
    print(f"📁 Detailed results saved in: {save_dir}")

if __name__ == '__main__':
    main()