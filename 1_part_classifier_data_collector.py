# ==========================================================
# 1. DATA COLLECTION
# ==========================================================
"""
In the first part, we go through subdirectories of anomaly_dataset listed below:
1. bracket_black
2. bracket_brown
3. bracket_white
4. metal_plate
Collect all the images with their corresponding labels (e.g., "bracket_black", "bracket_brown", etc.) into a single dataset.
Then use the data to train, validate, and test the model
"""
import os
import shutil
HEIGHT = 1024
WIDTH = 1024
import cv2
import numpy as np

class DataCollector:
    def __init__(self, anomaly_dataset,training_dataset_path, list_of_labels):
        self.training_dataset_path = training_dataset_path
        self.anomaly_dataset = anomaly_dataset
        #create the training_dataset_path_directory if non existing
        if not os.path.exists(self.training_dataset_path):
            os.makedirs(self.training_dataset_path)
        self.list_of_labels = list_of_labels
        for label in self.list_of_labels:
            label_path = os.path.join(self.training_dataset_path, label)
            label_ground_truth_path = os.path.join(self.training_dataset_path, label + "_ground_truth")
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            if not os.path.exists(label_ground_truth_path):
                os.makedirs(label_ground_truth_path)
        self.data = []

    def collect_data(self):
        for label in self.list_of_labels:
            label_path = os.path.join(self.anomaly_dataset, label)
            train_good_path = os.path.join(label_path, "train", "good")
            test_subfolders = os.path.join(label_path, "test")
            ground_truth_subfolders = os.path.join(label_path, "ground_truth")
            if os.path.exists(label_path):
                for file_name in os.listdir(train_good_path):
                    if file_name.endswith(".png"):
                        file_path = os.path.join(train_good_path, file_name)
                        self.data.append((file_path, label))
                        #copy file into training dataset with train_good in the beginning of the file name
                        new_file_path = os.path.join(self.training_dataset_path, label, "train_good_" + file_name)
                        shutil.copyfile(file_path, new_file_path)
                        #for each good train image, also create all black grayscale ground truth showing no defect and save it
                        # under ground truth folder
                        ground_truth_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                        ground_truth_path = os.path.join(self.training_dataset_path, label + "_ground_truth", "train_good_" + file_name)
                        cv2.imwrite(ground_truth_path, ground_truth_image)

                for subfolder in os.listdir(test_subfolders):
                    subfolder_path = os.path.join(test_subfolders, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file_name in os.listdir(subfolder_path):
                            if file_name.endswith(".png"):
                                file_path = os.path.join(subfolder_path, file_name)
                                self.data.append((file_path, label))
                                #copy file into training dataset with test_subfolder in the beginning of the file name
                                new_file_path = os.path.join(self.training_dataset_path, label, "test_" + subfolder + "_" + file_name)
                                shutil.copyfile(file_path, new_file_path)
                                if "good" in file_path:
                                    #for each good test image, also create all black grayscale ground truth showing no defect and save it
                                    # under ground truth folder
                                    ground_truth_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                                    ground_truth_path = os.path.join(self.training_dataset_path, label + "_ground_truth", "test_good" + "_" + file_name)
                                    cv2.imwrite(ground_truth_path, ground_truth_image)
                for subfolder in os.listdir(ground_truth_subfolders):
                    subfolder_path = os.path.join(ground_truth_subfolders, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file_name in os.listdir(subfolder_path):
                            if file_name.endswith(".png"):
                                file_path = os.path.join(subfolder_path, file_name)
                                self.data.append((file_path, label))
                                #copy file into training dataset with ground_truth_subfolder in the beginning of the file name
                                new_file_path = os.path.join(self.training_dataset_path, label + "_ground_truth", subfolder + "_" + file_name)
                                shutil.copyfile(file_path, new_file_path)
        return self.data

    

anomaly_dataset_path = "anomaly_dataset"
training_dataset_path = "training_dataset"
sample_data_collector = DataCollector(anomaly_dataset_path, training_dataset_path, ["bracket_black", "bracket_brown", "bracket_white", "metal_plate"])
sample_data_collector.collect_data()