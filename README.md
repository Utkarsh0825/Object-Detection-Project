# Object Detection Project with Faster R-CNN and OpenCV

This project implements an object detection system for detecting products on shelves using the Faster R-CNN model from PyTorch. It leverages OpenCV for image pre-processing and displays the results with bounding boxes around detected objects.

## Prerequisites

To run this project, you need to install the following libraries:

### Required Libraries

- **Python 3.6+**
- **PyTorch**: Deep learning framework used for the pre-trained Faster R-CNN model.
- **Torchvision**: A package containing the pre-trained Faster R-CNN model.
- **OpenCV**: Used for image processing tasks like resizing and normalizing.
- **PIL**: Python Imaging Library for handling images.
- **Matplotlib**: For visualizing images and bounding boxes.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Utkarsh0825/Object-Detection-Project.git
   cd Object-Detection-Project

2. Create and activate a virtual environment (optional but recommended):

bash
python3 -m venv object_detection_env
source object_detection_env/bin/activate  # macOS/Linux
object_detection_env\Scripts\activate     # Windows

**3. Install the required libraries:**

bash
pip install -r requirements.txt
Or manually install the necessary libraries:

bash
pip install torch torchvision opencv-python Pillow matplotlib

**Running the Code**
1.	Download a sample image of the shelf with products. You can place your image in the project directory, or specify the full path to the image. Make sure the image is in a compatible format (e.g., .jpg, .png).
Example: Download an image named shelf_image.jpg and place it in the project directory.
2.	Update the Image Path in the Code:
In the object_detection.py file, you'll need to specify the correct path for your image. Find the following line:
python
image_path = 'shelf_image.jpg'  # Path to your image
Replace 'shelf_image.jpg' with the full or relative path to the image you want to use.
Example:
python
image_path = '/Users/utkarsh/Documents/object_detection_project/shelf_image.jpg'
3.	Run the Script:
Once you've updated the image path, run the script using:
bash
python object_detection.py
The script will process the image and display the detected objects with bounding boxes around them.
 
**Output**

The script will display the image with detected objects outlined by green bounding boxes. It will also print out the class labels and their confidence scores for each detected object.


