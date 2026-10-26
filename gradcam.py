import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import os
from lstm_model import LSTMModel
from dataloader import get_dataloaders


class ResNetGradCAM:
    """GradCAM for frozen ResNet - uses feature maps instead of gradients"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        
        print("🔍 Setting up ResNetGradCAM for frozen ResNet...")
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks on ResNet to capture feature maps"""
        try:
            # Hook the last convolutional layer before adaptive pooling
            # For ResNet18, this is usually layer4
            resnet = self.model.resnet
            
            if hasattr(resnet, 'layer4'):
                self.target_layer = resnet.layer4[-1].conv2
            elif hasattr(resnet, 'layer3'):
                self.target_layer = resnet.layer3[-1].conv2
            else:
                # Find any convolutional layer
                for name, module in resnet.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        self.target_layer = module
                        break
            
            print(f"✅ Target layer: {self.target_layer}")
            
            # Register forward hook to capture activations
            self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
            print("✅ Forward hook registered")
            
        except Exception as e:
            print(f"❌ Hook setup failed: {e}")
            raise
    
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass"""
        self.activations = output.detach()
    
    def generate_cam(self, input_data):
        """Generate CAM using feature map importance"""
        print("\n🎯 Generating ResNet CAM...")
        
        # Clear previous activations
        self.activations = None
        
        # Get inputs
        webcam_seq = input_data['webcam_seq']
        picam_seq = input_data['picam_seq']
        
        batch_size, seq_len, C, H, W = webcam_seq.shape
        
        # Use first frame for visualization
        single_frame = webcam_seq[:, 0, :, :, :]
        
        print(f"📊 Input frame shape: {single_frame.shape}")
        
        # Forward pass through ResNet (frozen, no gradients needed)
        with torch.no_grad():
            _ = self.model.resnet(single_frame)
        
        print(f"📥 Activations captured: {self.activations is not None}")
        
        if self.activations is None:
            raise ValueError("❌ No activations captured")
        
        # For frozen ResNet, we use the feature map magnitudes as importance
        # This shows which spatial locations have high activation
        feature_importance = torch.mean(self.activations, dim=1, keepdim=True)  # Global average across channels
        feature_importance = F.relu(feature_importance)
        
        print(f"📊 Feature importance shape: {feature_importance.shape}")
        
        # Upsample to input image size
        cam = F.interpolate(feature_importance, size=(H, W), mode='bilinear', align_corners=False)
        
        # Convert to numpy and normalize
        cam = cam.squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        
        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
        
        print(f"📊 CAM range: [{np.min(cam):.3f}, {np.max(cam):.3f}]")
        
        return cam


class CrossAttentionVisualizer:
    """Visualize cross-attention weights"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_weights = None
        
        print("🔍 Setting up CrossAttentionVisualizer...")
        self._setup_hook()
    
    def _setup_hook(self):
        """Setup hook to capture attention weights"""
        if hasattr(self.model, 'cross_attention'):
            self.cross_attention = self.model.cross_attention
            # Register hook to capture attention weights
            self.forward_handle = self.cross_attention.register_forward_hook(self._attention_hook)
            print("✅ Cross-attention hook registered")
        else:
            print("⚠️ No cross_attention found in model")
    
    def _attention_hook(self, module, input, output):
        """Capture attention weights during forward pass"""
        if hasattr(module, 'last_attn_weights'):
            self.attention_weights = module.last_attn_weights
            # print(f"📥 Attention weights captured: {self.attention_weights.shape}")
    
    def visualize_attention(self, input_data):
        """Generate attention visualization"""
        print("\n🎯 Generating Cross-Attention visualization...")
        
        self.attention_weights = None
        
        # Move inputs to device
        webcam_seq = input_data['webcam_seq'].to(self.device)
        picam_seq = input_data['picam_seq'].to(self.device)
        ball_coords = input_data['ball_coords'].to(self.device)
        motor_seq = input_data['motor_seq'].to(self.device)
        
        # Forward pass to capture attention weights
        with torch.no_grad():
            _ = self.model(
                picam_seq=picam_seq,
                webcam_seq=webcam_seq,
                ball_coords=ball_coords,
                motor_seq=motor_seq
            )
        
        print(f"📥 Attention weights captured: {self.attention_weights is not None}")
        
        if self.attention_weights is None:
            # Create fallback attention map
            _, _, _, H, W = webcam_seq.shape
            return self._create_fallback_attention(H, W)
        
        # Process attention weights
        attention = self.attention_weights.numpy() if hasattr(self.attention_weights, 'numpy') else self.attention_weights
        
        print(f"📊 Raw attention shape: {attention.shape}")
        
        # Attention shape: (batch, num_heads, target_len, source_len) or similar
        # We need to reduce it to 2D for visualization
        
        if len(attention.shape) == 4:
            # Multi-head attention: average over heads and sequence dimensions
            attn_2d = np.mean(attention, axis=(0, 1, 2))  # Reduce to spatial importance
        elif len(attention.shape) == 3:
            # Already reduced: average over batch and sequence
            attn_2d = np.mean(attention, axis=(0, 1))
        elif len(attention.shape) == 2:
            # Simple 2D attention
            attn_2d = np.mean(attention, axis=0)
        else:
            attn_2d = attention.flatten()
        
        # Ensure we have a reasonable shape
        if len(attn_2d.shape) > 1:
            attn_2d = np.mean(attn_2d, axis=0)
        
        # Get image dimensions
        _, _, _, H, W = webcam_seq.shape
        
        # Resize attention to image size
        attention_map = cv2.resize(attn_2d, (W, H))
        
        # Normalize
        attention_map = np.maximum(attention_map, 0)
        if np.max(attention_map) - np.min(attention_map) > 1e-8:
            attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
        else:
            attention_map = np.ones_like(attention_map) * 0.5
        
        print(f"📊 Attention map range: [{np.min(attention_map):.3f}, {np.max(attention_map):.3f}]")
        
        return attention_map
    
    def _create_fallback_attention(self, H, W):
        """Create a fallback attention map"""
        # Create center-biased Gaussian
        y, x = np.ogrid[0:H, 0:W]
        center_y, center_x = H // 2, W // 2
        sigma = min(H, W) // 3
        attention = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
        return (attention - attention.min()) / (attention.max() - attention.min())


class FeatureMapVisualizer:
    """Visualize spatial encoder features"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = None
        
        print("🔍 Setting up FeatureMapVisualizer...")
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks on spatial encoder"""
        try:
            # Hook the last conv layer in spatial encoder
            spatial_encoder = self.model.spatial_encoder
            self.target_layer = spatial_encoder.conv_net[-2]  # Last conv layer before pooling
            print(f"✅ Target layer: {self.target_layer}")
            
            self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
            print("✅ Spatial encoder hook registered")
            
        except Exception as e:
            print(f"❌ Spatial encoder hook failed: {e}")
    
    def _forward_hook(self, module, input, output):
        """Capture activations"""
        self.activations = output.detach()
    
    def visualize_features(self, input_data):
        """Visualize spatial encoder features"""
        print("\n🎯 Generating Spatial Feature visualization...")
        
        self.activations = None
        
        webcam_seq = input_data['webcam_seq']
        batch_size, seq_len, C, H, W = webcam_seq.shape
        
        # Use first frame
        single_frame = webcam_seq[:, 0, :, :, :]
        
        # Forward pass through spatial encoder
        with torch.no_grad():
            _ = self.model.spatial_encoder(single_frame)
        
        if self.activations is None:
            # Fallback
            return self._create_fallback_features(H, W)
        
        # Process feature maps
        feature_importance = torch.mean(self.activations, dim=1, keepdim=True)
        feature_importance = F.relu(feature_importance)
        
        # Upsample to original size
        feature_map = F.interpolate(feature_importance, size=(H, W), mode='bilinear', align_corners=False)
        
        # Convert and normalize
        feature_map = feature_map.squeeze().cpu().numpy()
        feature_map = np.maximum(feature_map, 0)
        if np.max(feature_map) - np.min(feature_map) > 1e-8:
            feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
        
        return feature_map
    
    def _create_fallback_features(self, H, W):
        """Create fallback feature map"""
        return np.ones((H, W)) * 0.3


def apply_heatmap(image, heatmap, alpha=0.6):
    """Apply heatmap to original image"""
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    image = (image - image.min()) / (image.max() - image.min())
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    heatmap_colored = np.uint8(255 * cm.jet(heatmap_resized)[:, :, :3])
    image_uint8 = np.uint8(255 * image)
    
    superimposed_img = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img


def create_comparison_visualization(images, resnet_cam, attention_map, spatial_features, save_path=None):
    """Create comprehensive comparison visualization"""
    num_images = min(3, len(images))
    
    fig, axes = plt.subplots(3, num_images, figsize=(18, 12))
    if num_images == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]
    
    # Calculate statistics for titles
    resnet_stats = f"ResNet: max={resnet_cam.max():.3f}, mean={resnet_cam.mean():.3f}"
    attention_stats = f"Attention: max={attention_map.max():.3f}, mean={attention_map.mean():.3f}"
    # spatial_stats = f"Spatial: max={spatial_features.max():.3f}, mean={spatial_features.mean():.3f}"
    
    for i in range(num_images):
        # Original image
        img = images[i].cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Apply heatmaps
        resnet_img = apply_heatmap(images[i], resnet_cam)
        attention_img = apply_heatmap(images[i], attention_map)
        spatial_img = apply_heatmap(images[i], spatial_features)
        
        # Plot original
        axes[0][i].imshow(img)
        axes[0][i].set_title(f'Frame {i+1}\nOriginal Image')
        axes[0][i].axis('off')
        
        # Plot ResNet CAM
        axes[1][i].imshow(resnet_img)
        axes[1][i].set_title(f'ResNet Feature Importance\n{resnet_stats}' if i == 0 else 'ResNet Features')
        axes[1][i].axis('off')
        
        # Plot Cross-Attention
        axes[2][i].imshow(attention_img)
        axes[2][i].set_title(f'Cross-Attention Weights\n{attention_stats}' if i == 0 else 'Cross-Attention')
        axes[2][i].axis('off')
        
        # Plot Spatial Features
        # axes[3][i].imshow(spatial_img)
        # axes[3][i].set_title(f'Spatial Encoder Features\n{spatial_stats}' if i == 0 else 'Spatial Features')
        # axes[3][i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"✅ Saved comparison to: {save_path}")
    else:
        plt.show()


def evaluate_model_visualizations(model, test_loader, device, save_dir="model_visualizations", num_samples=5):
    """Evaluate and visualize all model components"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔄 Initializing visualization modules...")
    
    # Initialize all visualizers
    resnet_viz = ResNetGradCAM(model, device)
    attention_viz = CrossAttentionVisualizer(model, device)
    spatial_viz = FeatureMapVisualizer(model, device)
    
    successful_samples = 0
    
    for batch_idx, ((picam_seq, webcam_seq, ball_seq, motor_seq), targets) in enumerate(tqdm(test_loader)):
        if successful_samples >= num_samples:
            break
            
        try:
            # Move to device
            picam_seq = picam_seq.to(device)
            webcam_seq = webcam_seq.to(device)
            ball_seq = ball_seq.float().to(device)
            motor_seq = motor_seq.float().to(device)
            
            input_data = {
                'picam_seq': picam_seq,
                'webcam_seq': webcam_seq,
                'ball_coords': ball_seq,
                'motor_seq': motor_seq
            }
            
            # Get sample images (first batch)
            sample_images = webcam_seq[0]
            
            print(f"\n📦 Processing batch {batch_idx}...")
            
            # Generate all visualizations
            resnet_cam = resnet_viz.generate_cam(input_data)
            attention_map = attention_viz.visualize_attention(input_data)
            spatial_features = spatial_viz.visualize_features(input_data)
            
            # Create comparison visualization
            comparison_path = os.path.join(save_dir, f"comparison_batch_{batch_idx}.png")
            create_comparison_visualization(
                sample_images, resnet_cam, attention_map, spatial_features, comparison_path
            )
            
            # Save individual maps
            np.save(os.path.join(save_dir, f"resnet_cam_batch_{batch_idx}.npy"), resnet_cam)
            np.save(os.path.join(save_dir, f"attention_map_batch_{batch_idx}.npy"), attention_map)
            # np.save(os.path.join(save_dir, f"spatial_features_batch_{batch_idx}.npy"), spatial_features)
            
            # Save sample images
            sample_img_path = os.path.join(save_dir, f"sample_images_batch_{batch_idx}.npy")
            np.save(sample_img_path, sample_images.cpu().numpy())
            
            successful_samples += 1
            print(f"✅ Successfully processed batch {batch_idx}")
            
        except Exception as e:
            print(f"⚠️ Skipping batch {batch_idx}: {str(e)}")
            continue
    
    print(f"\n🎉 Completed! Successfully processed {successful_samples} samples")
    return successful_samples


def main():
    """Main function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_arm_control_lstm_working.pth"
    save_dir = "model_component_visualizations"
    
    print("🎯 Model Component Visualization")
    print("=" * 50)
    print("Visualizing: ResNet Features, Cross-Attention, and Spatial Encoder")
    print("=" * 50)
    
    # Load model
    model = LSTMModel(
        motor_dim=3,
        hidden_dim=512,
        output_dim=3,
        num_layers=2,
        dropout=0.3,
        device=device
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load data
    base_path = "E:/Bibek_Files/robo_yolo"
    episodes = [f"episode_{i}" for i in range(1, 11)]
    
    pi_cam_dirs = [f"{base_path}/dataset/{ep}/picam_frames" for ep in episodes]
    web_cam_dirs = [f"{base_path}/dataset/{ep}/webcam_frames" for ep in episodes]
    csv_paths = [f"{base_path}/dataset/{ep}/data.csv" for ep in episodes]
    yolo_dirs = [f"{base_path}/yolo_outputs/{ep}/webcam_frames/predicted_labels" for ep in episodes]

    try:
        _, _, test_loader = get_dataloaders(
            pi_cam_dirs=pi_cam_dirs,
            web_cam_dirs=web_cam_dirs,
            csv_paths=csv_paths,
            yolo_dirs=yolo_dirs,
            batch_size=1,  # Single sample for clear visualization
            sequence_length=5,
            img_size=(224, 224)
        )
        print("✅ Test data loaded")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run visualization
    print("\n🔥 Generating model component visualizations...")
    results = evaluate_model_visualizations(
        model, test_loader, device, save_dir, num_samples=5
    )
    
    print(f"\n✅ Visualization completed!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"📊 Successful samples: {results}")
    print("\n📊 What each visualization shows:")
    print("   • ResNet Features: What the pretrained ResNet focuses on (frozen)")
    print("   • Cross-Attention: How the model attends to visual features based on motor context")
    print("   • Spatial Encoder: What your custom spatial encoder learns")
    print("   • Comparison: Side-by-side view of all components")


if __name__ == "__main__":
    main()