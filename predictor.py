import torch
from torchvision import transforms
from PIL import Image
import os
from module import CNN_NeuralNet  # Import CNN_NeuralNet from the module file

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = CNN_NeuralNet(in_channels=3, num_diseases=10)  # Adjust parameters based on your model architecture
model = torch.load('leaf_pred.pth', map_location=device)
model.eval()

# Define the transform to be applied to the input image
transform = transforms.ToTensor()

def predict_image(image_path, model):
    """Predicts the class label for a given image."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = CNN_NeuralNet(in_channels=3, num_diseases=10)  # Adjust parameters based on your model architecture
    model = torch.load('leaf_pred.pth', map_location=device)
    model.eval()

    # Define the transform to be applied to the input image
    transform = transforms.ToTensor()
    # Open the image
    image = Image.open(image_path)
    # Apply the transformation
    image = transform(image).unsqueeze(0)
    # Move the tensor to the appropriate device
    image = image.to(device)
    # Get the predicted class probabilities
    with torch.no_grad():
        outputs = model(image)
    # Get the predicted class index
    _, predicted = torch.max(outputs, 1)
    # Return the predicted class index
    return predicted[0].item()

# Define the path of the image you want to predict
image_path = "F:\leaf\d\AppleCedarRust1.JPG" # Replace "image.jpg" with the filename of your image


# Predict the class label for the specified image
predicted_class = predict_image(image_path, model)
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

predicted_class_label = class_labels[predicted_class]
print(f'Image: {image_path}, Predicted Class: {predicted_class_label}')

