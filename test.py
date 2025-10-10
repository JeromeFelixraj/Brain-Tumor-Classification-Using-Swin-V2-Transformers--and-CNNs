import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import random
from tqdm import tqdm
from main import HybridSwinCNN

def main():
    torch.manual_seed(42)
    random.seed(42)

    MODEL_PATH = "model/model_epoch_7.pth"
    TEST_DIR = "dataset/testing"
    BATCH_SIZE = 8
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # GMNP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])

    # Force GMNP class order regardless of folder order
    class CustomImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            return classes, class_to_idx

    test_dataset = CustomImageFolder(TEST_DIR, transform=transform)
    print("Class mapping:", test_dataset.class_to_idx)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridSwinCNN(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", ncols=80):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = 100 * correct / len(all_labels)
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

    # Save report
    report_dir = f"test_report/report{random.randint(1000,9999)}"
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()

    # Sample predictions
    def display_sample_results():
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        indices = random.sample(range(len(test_dataset)), 5)
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            input_img = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_img)
                _, pred = torch.max(output, 1)
            axs[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
            color = 'green' if pred.item() == label else 'red'
            axs[i].set_title(f"T: {CLASS_NAMES[label]}\nP: {CLASS_NAMES[pred.item()]}", color=color)
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "sample_predictions.png"))
        plt.close()

    display_sample_results()

    print(f"\n📁 Report saved to: {report_dir}")

if __name__ == "__main__":
    main()
