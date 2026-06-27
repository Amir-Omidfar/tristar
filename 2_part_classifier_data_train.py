# ==========================================================
# 2. DATA PREPARATION (Updated for 70/20/10 Split)
# ==========================================================
"""
Using the sorted and collected data train a network for stage 1 classification task
"""
import os
import time
import copy
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

class ResNetStageOneModel():
    def __init__(self):
        # ==========================================================
        # 1. HYPERPARAMETERS & CONFIGURATION
        # ==========================================================
        self.DATASET_PATH = 'training_dataset'  # Your root data directory
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.LEARNING_RATE = 1e-4
        self.NUM_CLASSES = 4
        self.SAVE_PATH = 'best_stage1_resnet18.pth'
        
        # Device configuration (CUDA GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def prepare_data(self):
        # ==========================================================
        # 2. DATA PREPARATION
        # ==========================================================
        # Standard ResNet18 transformations with ImageNet normalization
        self.stage1_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

        if not os.path.exists(self.DATASET_PATH):
            raise FileNotFoundError(f"Dataset path '{self.DATASET_PATH}' not found. Please verify the folder directory.")

        # Load from structured folders
        self.full_dataset = datasets.ImageFolder(root=self.DATASET_PATH, transform=self.stage1_transforms)
        print(f"Class mapping found: {self.full_dataset.class_to_idx}")
        self.class_names = self.full_dataset.classes

        # Stratified-style random split (80% Train, 20% Val)
        self.train_size = int(0.7 * len(self.full_dataset))
        self.val_size = int(0.2 * len(self.full_dataset))
        self.test_size = len(self.full_dataset) - self.train_size - self.val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [self.train_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        print(f"Loaded {len(self.train_dataset)} training images, {len(self.val_dataset)} validation images, and {len(self.test_dataset)} test images.\n")

    def setup_model_params(self):
        # ==========================================================
        # 3. MODEL, LOSS FUNCTION, & OPTIMIZER
        # ==========================================================
        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        # Replace the final fully connected layer for our 4 specific classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.NUM_CLASSES)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def train(self):
        # ==========================================================
        # 4. TRAINING LOOP WITH VALIDATION
        # ==========================================================
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_val_loss = float('inf')

        since = time.time()

        for epoch in range(1, self.EPOCHS + 1):
            print(f"Epoch {epoch}/{self.EPOCHS}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = self.train_loader
                else:
                    self.model.eval()   # Set model to evaluation mode
                    dataloader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data batches
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero out parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass tracking history only if in train phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward pass & optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics tracking
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Deep copy the model weights if this is the best validation loss so far
                if phase == 'val' and epoch_loss < self.best_val_loss:
                    self.best_val_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    print(f"--> Found better model weights! Saving checkpoint...")
            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        # Load best model weights and save to disk
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.model.state_dict(), self.SAVE_PATH)
        print(f"Best model weights successfully saved to '{self.SAVE_PATH}'")

    def evaluate_model_on_test_set(self):
        # ==========================================================
        # 5. TEST SET EVALUATION
        # ==========================================================
        print("=" * 60)
        print("Evaluating model on the completely unseen test set...")
        print("=" * 60)
        
        # Ensure the model uses the best saved weights and is in evaluation mode
        if os.path.exists(self.SAVE_PATH):
            self.model.load_state_dict(torch.load(self.SAVE_PATH, map_location=self.device))
        
        self.model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Generate metrics report
        report = classification_report(all_labels, all_preds, target_names=self.class_names)
        print("\nCLASSIFICATION REPORT:")
        print(report)

        # Generate pretty confusion matrix
        print("=" * 60)
        print("CONFUSION MATRIX:")
        print("=" * 60)
        cm = confusion_matrix(all_labels, all_preds)
        
        header = f"{'True \ Pred':<16}" + "".join([f"{name[:13]:<15}" for name in self.class_names])
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(cm):
            row_str = f"{self.class_names[i][:13]:<16}" + "".join([f"{val:<15}" for val in row])
            print(row_str)
        print("=" * 60)

resnet_model = ResNetStageOneModel()
resnet_model.prepare_data()
resnet_model.setup_model_params()
#resnet_model.train()
resnet_model.evaluate_model_on_test_set()