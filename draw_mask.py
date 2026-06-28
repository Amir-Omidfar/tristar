def draw_mask_on_images(output_dir="mask_visualizations"):
        import os
        import cv2
        import numpy as np

        # Create an output directory so you don't clutter your dataset folder
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir("training_dataset/bracket_white"):
            # Skip any system files or directories
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            mask_name = file_name.replace(".png", "_mask.png")
            mask_path = "training_dataset/bracket_white_ground_truth/" + mask_name
            
            img = cv2.imread("training_dataset/bracket_white/" + file_name)
            if img is None:
                continue

            # Check if the ground truth mask file exists (good parts won't have one)
            if os.path.exists(mask_path):
                # Read mask in grayscale
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    # Create a solid color canvas matching the image size (Bright Red in BGR)
                    color_overlay = np.zeros_like(img)
                    color_overlay[mask > 0] = [0, 0, 255]  # [B, G, R]

                    # Blend the original image and the color overlay together
                    # alpha=0.75 (original sharpness), beta=0.25 (transparency of mask color)
                    img_visualized = cv2.addWeighted(img, 0.75, color_overlay, 0.25, 0)
                else:
                    img_visualized = img
            else:
                # If no mask exists, it's a clean/good part; save it as-is to verify
                img_visualized = img

            # Save the final diagnostic frame
            save_path = os.path.join(output_dir, f"visualized_{file_name}")
            cv2.imwrite(save_path, img_visualized)
            
        print(f"🎯 Ground truth overlay generation complete! Images saved to: ./{output_dir}")

draw_mask_on_images()