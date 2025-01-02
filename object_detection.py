import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

def preprocess_image(image_path):
    """Load and preprocess the image for detection."""
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def detect_objects(image_tensor):
    """Detect objects in the image using the pre-trained model."""
    with torch.no_grad():  # Disable gradient computation
        predictions = model(image_tensor)
    return predictions

def postprocess_predictions(predictions, confidence_threshold=0.5):
    """Post-process the model's predictions."""
    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()

    valid_indices = np.where(scores > confidence_threshold)
    return boxes[valid_indices], labels[valid_indices], scores[valid_indices]

def draw_bounding_boxes(image_path, boxes, labels):
    """Draw bounding boxes on the image."""
    image = cv2.imread(image_path)
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path):
    """Main function to run object detection."""
    image_tensor = preprocess_image(image_path)
    predictions = detect_objects(image_tensor)
    boxes, labels, scores = postprocess_predictions(predictions)
    draw_bounding_boxes(image_path, boxes, labels)

if __name__ == '__main__':
    image_path = '/Users/utkarsh/Downloads/shelf_image.jpg'  # Path to the image you want to process
    main(image_path)
