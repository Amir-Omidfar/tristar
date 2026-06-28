"""
Using the synchronized images and masks to train a U-Net for Stage 2 pixel-level defect inspection.
"""
import os
import time
import copy
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import numpy as np
import cv2

# ==========================================================
# 1. CUSTOM SYNCHRONIZED DATASET
# ==========================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # Pull and sort image paths to align exactly with mask files
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) + 
                                  glob.glob(os.path.join(image_dir, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, filename)
        
        if not os.path.exists(mask_path):
            #change filename to have a _mask suffix
            mask_path = os.path.join(self.mask_dir, filename.replace(".png", "_mask.png"))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing ground truth mask match for image: {filename}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        
        # Synchronized Resizing
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Synchronized Spatial Augmentation (Flips keep labels aligned perfectly)
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Separate Tensor Transformations
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask_np = np.array(mask)
        # If ANY pixel is greater than 0, flag it as a 1.0 float defect matrix layer
        mask_bin = np.zeros_like(mask_np, dtype=np.float32)
        mask_bin[mask_np > 0] = 1.0
        
        # Convert directly to a tensor and add the explicit channel dimension [1, H, W]
        mask = torch.from_numpy(mask_bin).unsqueeze(0)
        # ------------------------

        return image, mask

# ==========================================================
# 2. COMBINED ROBUST LOSS FUNCTION (BCE + Dice)
# ==========================================================
class BCEWithDiceLoss(nn.Module):
    def __init__(self):
        super(BCEWithDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)

#Dice Loss + Focal Loss
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, dice_weight=1.0, focal_weight=1.0, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha          # Focal loss balancing parameter
        self.gamma = gamma          # Focal loss focusing parameter (higher = focuses more on hard defects)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure correct shapes for calculation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()
        
        # Convert logits to probabilities safely
        probs = torch.sigmoid(y_pred)
        probs = torch.clamp(probs, self.smooth, 1.0 - self.smooth)
        
        # ----------------------------------------------------------
        # 1. BINARY FOCAL LOSS CALCULATION
        # ----------------------------------------------------------
        # Penalizes the model heavily when it confidently guesses "background (0)" for a true defect (1)
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        p_t = probs * y_true + (1 - probs) * (1 - y_true)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * bce
        focal_loss = focal_loss.mean()

        # ----------------------------------------------------------
        # 2. DICE LOSS CALCULATION
        # ----------------------------------------------------------
        # Measures overlap directly to ensure small white-on-white scratches don't get ignored
        intersection = (probs * y_true).sum()
        dice_coef = (2. * intersection + self.smooth) / (probs.sum() + y_true.sum() + self.smooth)
        dice_loss = 1.0 - dice_coef

        # Combined loss weighted appropriately
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)

# ==========================================================
# 3. SEGMENTATION TRAINING MANAGER CLASS
# ==========================================================
class UNetSegmentationModel:
    def __init__(self,img_dir,mask_dir,part_name):
        # Configuration parameters
        self.IMAGE_DIR = img_dir
        self.MASK_DIR = mask_dir
        self.BATCH_SIZE = 16                                # Lower batch size since segmentation takes more VRAM
        self.EPOCHS = 15
        self.LEARNING_RATE = 2e-4
        self.SAVE_PATH = 'best_stage2_unet_resnet34_'+part_name+'.pth'
        self.part_name = part_name
        self.training_data_bad_samples_multiplier = {
            "bracket_black": 6,
            "bracket_brown": 4,
            "bracket_white": 4,
            "metal_plate": 1
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for Stage 2: {self.device}")

    def count_samples(self):
        good_samples = []
        bad_samples = []
        for image_file_name in os.listdir(self.IMAGE_DIR):
            if "train_good" in image_file_name or "test_good" in image_file_name:
                good_samples.append(image_file_name)
            else:
                bad_samples.append(image_file_name)
        print(f"Good samples: {len(good_samples)}, Bad samples: {len(bad_samples)}")

    def prepare_data(self):
        full_dataset = SegmentationDataset(self.IMAGE_DIR, self.MASK_DIR)
        total_size = len(full_dataset)
        
        # 70/20/10 Split to remain completely consistent with Stage 1
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2)

        print(f"Loaded {len(self.train_dataset)} train, {len(self.val_dataset)} val, and {len(self.test_dataset)} test pairs.")

    def prepare_data_aug(self):
        # 1. Base initialization from your raw data folders
        full_dataset = SegmentationDataset(self.IMAGE_DIR, self.MASK_DIR)
        total_size = len(full_dataset)
        
        # Standard structural split
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        # Split using a fixed seed to get stable raw assignments
        base_train, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 2. OVERSAMPLE ONLY WITHIN THE TRAINING SET TO PREVENT LEAKAGE
        balanced_train_indices = []
        
        for idx in base_train.indices:
            img_path = full_dataset.image_paths[idx]
            
            # Check if this training path belongs to a bad sample
            if "train_good" not in img_path and "test_good" not in img_path:
                # Multiply this bad index instance inside the training pool to match weights
                #balanced_train_indices.extend([idx] * 4)
                balanced_train_indices.extend([idx] * self.training_data_bad_samples_multiplier.get(self.part_name, 4))
            else:
                balanced_train_indices.append(idx)
                
        # Create a customized subset using our oversampled indices
        self.train_dataset = torch.utils.data.Subset(full_dataset, balanced_train_indices)
        
        # 3. Build DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        
        print(f"Oversampled Training Pool Size: {len(self.train_dataset)}")
        print(f"Validation Pool Size: {len(self.val_dataset)}")
        print(f"Strict Unseen Test Pool Size: {len(self.test_dataset)}")
    
    def setup_model(self):
        # Create U-Net with a pre-trained ResNet34 encoder
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1, # Single output channel representing defect mask probability
            activation=None # We pass raw logits into BCEWithDiceLoss for numerical stability
        ).to(self.device)

        self.criterion = BCEWithDiceLoss()
        if self.part_name == "bracket_white":
            print("="*30)
            print("⚠️ Using Dice + Focal Loss for bracket_white due to white sea pixels")
            print("="*30)
            self.criterion = DiceFocalLoss()  # Use Dice + Focal for bracket_white due to class imbalance
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def train(self):
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_val_loss = float('inf')
        since = time.time()

        for epoch in range(1, self.EPOCHS + 1):
            print(f"Epoch {epoch}/{self.EPOCHS}\n" + "-"*10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.train_loader
                else:
                    self.model.eval()
                    dataloader = self.val_loader

                running_loss = 0.0

                for inputs, masks in dataloader:
                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, masks)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloader.dataset)
                print(f"{phase.capitalize()} Combined Loss: {epoch_loss:.4f}")

                # Tracking validation loss optimization checkpoint
                if phase == 'val' and epoch_loss < self.best_val_loss:
                    self.best_val_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    print("--> Found better segmentation weights! Saving checkpoint...")
            print()

        time_elapsed = time.time() - since
        print(f"Stage 2 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best Val Loss achieved: {self.best_val_loss:.4f}")

        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), self.SAVE_PATH)
        print(f"Weights written out completely to: '{self.SAVE_PATH}'")

    def evaluate_model_on_test_set(self):
        # ==========================================================
        # 5. TEST SET EVALUATION & VISUALIZATION
        # ==========================================================
        print("=" * 60)
        print("Running Stage 2 Evaluation on Unseen Test Split...")
        print("=" * 60)
        
        # Ensure the best saved weights are loaded and model is in eval mode
        if os.path.exists(self.SAVE_PATH):
            self.model.load_state_dict(torch.load(self.SAVE_PATH, map_location=self.device))
        
        self.model.eval()
        
        total_iou = 0.0
        num_samples = 0
        plot_count = 0
        max_visualizations = 5 # Number of sample images to save to disk
        
        # Import plotting tools locally to keep dependencies clean
        import matplotlib.pyplot as plt
        import numpy as np

        with torch.no_grad():
            for inputs, masks in self.test_loader:
                inputs_dev = inputs.to(self.device)
                masks_dev = masks.to(self.device)
                
                outputs = self.model(inputs_dev)
                
                # Convert raw network outputs (logits) to a probability map [0, 1]
                pred_masks = torch.sigmoid(outputs)
                # Binarize predictions using standard 0.5 confidence threshold
                pred_masks_bin = (pred_masks > 0.5).float()

                # Loop through the batch to compute metrics and handle plotting
                for i in range(inputs.size(0)):
                    p_mask = pred_masks_bin[i].cpu()
                    t_mask = masks[i].cpu()
                    
                    # Compute Intersection over Union (IoU) for this image
                    intersection = (p_mask * t_mask).sum()
                    union = p_mask.sum() + t_mask.sum() - intersection
                    
                    # Handle edge case where image is clean (no defects) and model correctly predicts no defects
                    if union == 0:
                        iou = 1.0 
                    else:
                        iou = (intersection / union).item()
                        
                    total_iou += iou
                    num_samples += 1
                    
                    # Save a few side-by-side visualization figures for your GitHub repo
                    if plot_count < max_visualizations:
                        orig_img = inputs[i].numpy().transpose(1, 2, 0)
                        # Denormalize image using standard ImageNet stats to display true colors
                        orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        orig_img = np.clip(orig_img, 0, 1)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(orig_img)
                        axes[0].set_title("Raw Input Image")
                        axes[0].axis('off')
                        
                        axes[1].imshow(t_mask.squeeze(), cmap='gray')
                        axes[1].set_title("True Defect Mask")
                        axes[1].axis('off')
                        
                        axes[2].imshow(p_mask.squeeze(), cmap='gray')
                        axes[2].set_title(f"Predicted Mask (IoU: {iou:.2f})")
                        axes[2].axis('off')

                        output_img_name = f"stage2_test_sample_{self.part_name}_{plot_count}.png"
                        plt.tight_layout()
                        plt.savefig(output_img_name, bbox_inches='tight')
                        plt.close()
                        print(f"Saved visual prediction report to: {output_img_name}")
                        plot_count += 1

        mean_iou = total_iou / num_samples
        print("\n" + "=" * 60)
        print(f"STAGE 2 PERFORMANCE METRIC")
        print("-" * 60)
        print(f"Mean Intersection over Union (mIoU) on Test Set: {mean_iou:.4f}")
        print("=" * 60)
        return 

     
if __name__ == "__main__":
    unet_pipeline = UNetSegmentationModel("training_dataset/bracket_white","training_dataset/bracket_white_ground_truth","bracket_white")
    #unet_pipeline.count_samples()
    #unet_pipeline.count_samples()
    unet_pipeline.prepare_data_aug()
    unet_pipeline.setup_model()
    unet_pipeline.train()
    unet_pipeline.evaluate_model_on_test_set()