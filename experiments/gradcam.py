import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augmentations = [
    "NoneVerdadeiro", "Flipping", "Shift", "RandomErasing",
    "AutoAugment", "RandAugment", "TrivialAugment", "AugMix"
]

model_name = "resnet50"
num_classes = 8
weights_dir = "model_weights"
output_dir = "gradcam_comparisons"
os.makedirs(output_dir, exist_ok=True)

# Mapping class index to class name
classes = {
    0: "10",
    1: "12",
    2: "14",
    3: "16",
    4: "18",
    5: "6",
    6: "8",
    7: "ctrl"
}

# example image per class
image_paths = {
    0: "../dataset/orig/10/10_A_002.TIF",
    1: "../dataset/orig/12/12_A_002.TIF",
    2: "../dataset/orig/14/14_A_002.TIF",
    3: "../dataset/orig/16/16_A_002.TIF",
    4: "../dataset/orig/18/18_A_002.TIF",
    5: "../dataset/orig/6/6_A_002.TIF",
    6: "../dataset/orig/8/8_A_002.TIF",
    7: "../dataset/orig/ctrl/CTRL_ZERO_002.TIF"
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def gradcam_analysis(model_name, model_paths, image_paths, classes, device, output_dir):
    """
    Generate Grad-CAM visualizations for multiple augmentations of a single model.
    """

    for _, img_path in image_paths.items():
        img_name = os.path.basename(img_path).split(".")[0]
        img = Image.open(img_path).convert("RGB")

        # list of images to plot (original + GradCAMs)
        cams_to_plot = [("Original Image", np.array(img.resize((224,224)))/255.0)]

        # loop through augmentations
        for aug in augmentations:
            # load pretrained model
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            model_path = model_paths.get(aug)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # input tensor
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            # GradCAM 
            target_layers = [model.layer4[-1]]  # last layer
            cam = GradCAM(model=model, target_layers=target_layers)

            # predicted class
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = output.argmax(dim=1).item()

            # Compute GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])
            grayscale_cam = grayscale_cam[0, :]

            # Visualization
            rgb_img = np.array(img.resize((224,224))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cams_to_plot.append((aug, visualization))

        # Plot all augmentations 
        n_cols = len(cams_to_plot)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))

        for ax, (aug_type, vis) in zip(axes, cams_to_plot):
            ax.imshow(vis)
            ax.set_title(f"{aug_type}", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{img_name}_gradcam_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved Grad-CAM comparison: {save_path}")


if __name__ == "__main__":
    model_paths = {aug: os.path.join(weights_dir, f"best_model_{model_name}_{aug}.pt")
                   for aug in augmentations}

    gradcam_analysis(model_name, model_paths, image_paths, classes, device, output_dir)
