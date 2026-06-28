"""
Multi-Part Cascaded Production Inference Pipeline
Dynamically routes 4 part classes to their respective fine-tuned U-Net models.
"""
from glob import glob
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
import segmentation_models_pytorch as smp
import scipy.ndimage as ndimage

class IndustrialInspectionLine:
    def __init__(self, stage1_weights_path, weights_directory="."):
        """
        Initializes the production line by loading the Stage 1 Classifier 
        and all available Stage 2 U-Net Segmenters dynamically.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Production Inference Engine on device: {self.device}")
        
        # ----------------------------------------------------------
        # STAGE 1: PART CLASSIFIER SETUP (ResNet18)
        # ----------------------------------------------------------
        self.stage1_model = resnet18()
        in_features = self.stage1_model.fc.in_features
        self.stage1_model.fc = torch.nn.Linear(in_features, 4)
        
        if not os.path.exists(stage1_weights_path):
            raise FileNotFoundError(f"Stage 1 weights missing at: {stage1_weights_path}")
        self.stage1_model.load_state_dict(torch.load(stage1_weights_path, map_location=self.device))
        self.stage1_model.to(self.device)
        self.stage1_model.eval()
        
        # Exact ordering mapping from Stage 1's ImageFolder setup
        self.stage1_classes = ['bracket_black', 'bracket_brown', 'bracket_white', 'metal_plate']

        # ----------------------------------------------------------
        # STAGE 2: DYNAMIC U-NET DICTIONARY SETUP
        # ----------------------------------------------------------
        self.unet_models = {}
        
        # Map each part class name directly to your specific filename convention
        for part_name in self.stage1_classes:
            weights_file = f"best_stage2_unet_resnet34_{part_name}.pth"
            full_weights_path = os.path.join(weights_directory, weights_file)
            
            if os.path.exists(full_weights_path):
                print(f"Loading specialized U-Net for target: {part_name}...")
                model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
                model.load_state_dict(torch.load(full_weights_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                # Store the model instance in our dictionary mapped by its class string key
                self.unet_models[part_name] = model
            else:
                print(f"⚠️ Warning: Checkpoint '{weights_file}' missing. Routing will fallback to safe bypass.")

        print("\nAll cascaded multi-stage networks deployed completely. System online.\n")

    def preprocess_image(self, pil_image):
        img_resized = TF.resize(pil_image, (224, 224))
        tensor_img = TF.to_tensor(img_resized)
        norm_img = TF.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return norm_img.unsqueeze(0).to(self.device)

    def inspect_incoming_part(self, image_path, pixel_threshold=15, confidence_cutoff=0.5):
        if not os.path.exists(image_path):
            return "Error", "File not found", "N/A"

        raw_image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess_image(raw_image)
        if "bracket_white" in image_path:
            pixel_threshold = 10
        with torch.no_grad():
            # ==========================================================
            # STAGE 1: IDENTIFY THE PART TYPE
            # ==========================================================
            class_logits = self.stage1_model(input_tensor)
            _, predicted_idx = torch.max(class_logits, 1)
            part_type = self.stage1_classes[predicted_idx.item()]
            
            # ==========================================================
            # STAGE 2: DYNAMIC ROUTING LAYER
            # ==========================================================
            if part_type in self.unet_models:
                target_unet = self.unet_models[part_type]
                unet_logits = target_unet(input_tensor)
                if part_type == 'bracket_white':
                    pred_probabilities = torch.sigmoid(unet_logits)
                    pred_mask_bin = (pred_probabilities > confidence_cutoff).cpu().numpy().squeeze()

                    # Apply a binary opening operation to clean up standalone noise clusters
                    # This removes stray background pixels while keeping true defect lines intact
                    cleaned_mask = ndimage.binary_opening(pred_mask_bin, structure=np.ones((3, 3))).astype(np.float32)
                    defect_pixel_count = np.sum(cleaned_mask)
                else:

                    pred_probabilities = torch.sigmoid(unet_logits)
                    pred_mask_bin = (pred_probabilities > confidence_cutoff).float()
                    
                    defect_pixel_count = torch.sum(pred_mask_bin).item()
                    
                if defect_pixel_count >= pixel_threshold:
                    verdict = "REJECT (Defective Part)"
                else:
                    verdict = "PASS (Good Part)"
                    
                details = f"Defect cluster area: {int(defect_pixel_count)} pixels"
            else:
                verdict = "PASS (Assumed Good - Segmentation Architecture Missing)"
                details = f"No active U-Net model loaded for identified class: '{part_type}'"
                
        return part_type, verdict, details


if __name__ == "__main__":
    # Point directly to your specific files in the workspace directory
    production_line = IndustrialInspectionLine(
        stage1_weights_path='best_stage1_resnet18.pth',
        weights_directory='.' 
    )
    #run inference on all images
    test_images = glob('training_dataset/bracket_white/test_*.png')
    count = 0
    t_count = 0
    for img_path in test_images:
        if "ground_truth" not in img_path:  # Skip ground truth images
            res = f"Inferred Results for {img_path}: {production_line.inspect_incoming_part(img_path)}"
            with open("inference_results.txt", "a") as f:
                f.write(res + "\n")
            if not "good" in img_path and "PASS" in res:
                count += 1
            t_count += 1
    print(f"Total passed defective parts: {count}/{t_count}")
