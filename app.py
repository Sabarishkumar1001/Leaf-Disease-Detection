import os
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Import the CNN_NeuralNet class from your module
from module import CNN_NeuralNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = CNN_NeuralNet(in_channels=3, num_diseases=38)  # Adjust parameters based on your model architecture
model=torch.load('leaf_pred.pth', map_location=device)
model.to(device)
model.eval()

# Define the transform to be applied to the input image
transform = transforms.ToTensor()

def predict_image(image_path, model):
    """Predicts the class label for a given image."""
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

# Class labels
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

precautionary_measures = {
     'Apple___Apple_scab': [
                "Prune and dispose of infected plant parts.",
                "Apply fungicides labeled for apple scab control, following label instructions.",
                "Maintain good air circulation around plants to reduce humidity."
            ],
            'Apple___Black_rot': [
                "Prune infected plant parts and remove fallen leaves and fruit promptly.",
                "Apply fungicides labeled for black rot control, especially during wet weather conditions.",
                "Ensure proper spacing between plants to improve air circulation."
            ],
            'Apple___Cedar_apple_rust': [
                "Plant resistant apple varieties if available.",
                "Remove cedar trees (Juniperus species) near apple orchards to reduce spore production.",
                "Apply fungicides preventatively, especially during spring when symptoms typically appear."
            ],
            'Apple___healthy': [
                "Practice good orchard sanitation, including removal of fallen leaves and fruit.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected."
            ],
            'Blueberry___healthy': [
                "Prune bushes to improve air circulation and light penetration.",
                "Control weeds and remove debris to reduce disease pressure.",
                "Apply organic mulch to conserve moisture and suppress weeds."
            ],
            'Cherry_(including_sour)___Powdery_mildew': [
                "Prune infected plant parts and remove debris from around plants.",
                "Apply fungicides labeled for powdery mildew control, especially during periods of high humidity.",
                "Avoid overhead irrigation to reduce leaf wetness."
            ],
            'Cherry_(including_sour)___healthy': [
                "Plant disease-resistant cherry varieties when available.",
                "Provide adequate spacing between plants to improve air circulation.",
                "Monitor for pests and diseases regularly and take appropriate action if detected."
            ],
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
                "Rotate crops with non-host plants to reduce disease buildup in the soil.",
                "Apply fungicides labeled for gray leaf spot control, following label instructions.",
                "Practice proper irrigation to avoid excessive leaf wetness."
            ],
            'Corn_(maize)___Common_rust_': [
                "Plant resistant corn varieties when available.",
                "Monitor weather conditions and apply fungicides preventatively during periods of high humidity and warm temperatures.",
                "Remove volunteer corn plants and weed hosts to reduce disease inoculum."
            ],
            'Corn_(maize)___Northern_Leaf_Blight': [
                "Rotate crops with non-host plants and practice good field sanitation.",
                "Apply fungicides preventatively, especially during periods of warm, humid weather.",
                "Choose corn hybrids with genetic resistance to northern leaf blight if available."
            ],
            'Corn_(maize)___healthy': [
                "Maintain proper crop rotation practices.",
                "Ensure adequate soil fertility and pH balance.",
                "Monitor for pests and diseases regularly and take appropriate action if detected."
            ],
            'Grape___Black_rot': [
                "Prune and destroy infected plant parts.",
                "Apply fungicides labeled for black rot control, especially during wet weather.",
                "Maintain good air circulation around vines."
            ],
            'Grape___Esca_(Black_Measles)': [
                "Remove and destroy infected vines.",
                "Avoid wounding vines during pruning and other vineyard operations.",
                "Apply fungicides labeled for esca control, although effectiveness may be limited."
            ],
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
                "Prune and destroy infected leaves and shoots.",
                "Apply fungicides labeled for leaf blight control.",
                "Maintain good air circulation and reduce vineyard humidity."
            ],
            'Grape___healthy': [
                "Practice good vineyard sanitation, including removal of fallen leaves and fruit.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected.",
                "Ensure proper irrigation and fertilization practices."
            ],

            'Orange___Haunglongbing_(Citrus_greening)': [
                "Plant citrus trees from certified disease-free stock.",
                "Regularly inspect trees for signs of disease and remove infected trees promptly.",
                "Control Asian citrus psyllid populations using insecticides."
            ],
            'Peach___Bacterial_spot': [
                "Apply copper-based bactericides during the growing season.",
                "Remove and destroy infected leaves and fruit.",
                "Plant resistant varieties if available."
            ],
            'Peach___healthy': [
                "Maintain good orchard sanitation, including removal of fallen leaves and fruit.",
                "Ensure proper irrigation and fertilization practices.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected."
            ],
            'Pepper,_bell___Bacterial_spot': [
                "Use disease-free seed and transplants.",
                "Apply copper-based bactericides during the growing season.",
                "Rotate crops with non-host plants to reduce disease buildup in the soil."
            ],
            'Pepper,_bell___healthy': [
                "Maintain proper crop rotation practices.",
                "Ensure adequate soil fertility and pH balance.",
                "Monitor for pests and diseases regularly and take appropriate action if detected."
            ],

            'Potato___Early_blight': [
                "Apply fungicides labeled for early blight control, following label instructions.",
                "Remove and destroy infected plant debris and volunteer plants.",
                "Rotate crops with non-host plants to reduce disease buildup in the soil."
            ],
            'Potato___Late_blight': [
                "Apply fungicides preventatively, especially during periods of wet weather.",
                "Remove and destroy infected plants and tubers promptly.",
                "Ensure proper spacing and good air circulation to reduce humidity."
            ],
            'Potato___healthy': [
                "Practice good crop rotation and avoid planting potatoes in the same area year after year.",
                "Maintain proper soil fertility and moisture levels.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected."
            ],
            'Raspberry___healthy': [
                "Prune bushes to improve air circulation and light penetration.",
                "Control weeds and remove debris to reduce disease pressure.",
                "Apply organic mulch to conserve moisture and suppress weeds."
            ],
            'Soybean___healthy': [
                "Rotate crops with non-host plants to break the disease cycle.",
                "Ensure proper spacing between plants to improve air circulation.",
                "Monitor for pests and diseases regularly and take appropriate action if detected."
            ],

            'Squash___Powdery_mildew': [
                "Prune infected plant parts and remove debris from around plants.",
                "Apply fungicides labeled for powdery mildew control, especially during periods of high humidity.",
                "Avoid overhead irrigation to reduce leaf wetness."
            ],
            'Strawberry___Leaf_scorch': [
                "Remove and destroy infected leaves and plant debris.",
                "Apply fungicides labeled for leaf scorch control, following label instructions.",
                "Ensure proper spacing and air circulation around plants."
            ],
            'Strawberry___healthy': [
                "Practice good crop rotation and avoid planting strawberries in the same area year after year.",
                "Maintain proper soil fertility and moisture levels.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected."
            ],
            'Tomato___Bacterial_spot': [
                "Remove and destroy infected plants and plant debris.",
                "Avoid overhead watering to reduce leaf wetness.",
                "Apply copper-based bactericides preventatively, following label instructions."
            ],
            'Tomato___Early_blight': [
                "Remove and destroy infected leaves and plant debris.",
                "Apply fungicides labeled for early blight control, especially during periods of wet weather.",
                "Practice crop rotation and avoid planting tomatoes in the same area year after year."
            ],
            'Tomato___Late_blight': [
                "Apply fungicides preventatively, especially during periods of wet weather.",
                "Remove and destroy infected plants and tubers promptly.",
                "Ensure proper spacing and good air circulation to reduce humidity."
            ],
            'Tomato___Leaf_Mold': [
                "Prune and remove infected leaves and plant debris.",
                "Apply fungicides labeled for leaf mold control, following label instructions.",
                "Improve air circulation around plants and reduce humidity."
            ],
            'Tomato___Septoria_leaf_spot': [
                "Remove and destroy infected leaves and plant debris.",
                "Apply fungicides labeled for Septoria leaf spot control, following label instructions.",
                "Ensure proper spacing and air circulation around plants."
            ],
            'Tomato___Spider_mites Two-spotted_spider_mite': [
                "Spray plants with a strong stream of water to dislodge mites.",
                "Apply insecticidal soap or miticides labeled for spider mite control.",
                "Maintain proper watering and avoid drought stress on plants."
            ],
            'Tomato___Target_Spot': [
                "Remove and destroy infected leaves and plant debris.",
                "Apply fungicides labeled for target spot control, following label instructions.",
                "Ensure proper spacing and air circulation around plants."
            ],
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
                "Remove and destroy infected plants promptly.",
                "Control whitefly populations, as they are the primary vector of the virus.",
                "Use resistant tomato varieties if available."
            ],
            'Tomato___Tomato_mosaic_virus': [
                "Remove and destroy infected plants and plant debris.",
                "Practice good sanitation, including washing hands and tools after handling infected plants.",
                "Use virus-free seeds and resistant tomato varieties if available."
            ],
            'Tomato___healthy': [
                "Practice good crop rotation and avoid planting tomatoes in the same area year after year.",
                "Maintain proper soil fertility and moisture levels.",
                "Monitor for signs of pests and diseases regularly and take appropriate action if detected."
            ]


}

class1_labels = {
    'Apple___Apple_scab': 'Apple - Apple scab',
    'Apple___Black_rot': 'Apple - Black rot',
    'Apple___Cedar_apple_rust': 'Apple - Cedar apple rust',
    'Apple___healthy': 'Apple - Healthy',
    'Blueberry___healthy': 'Blueberry - Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry (including sour) - Powdery mildew',
    'Cherry_(including_sour)___healthy': 'Cherry (including sour) - Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn (maize) - Cercospora leaf spot Gray leaf spot',
    'Corn_(maize)___Common_rust_': 'Corn (maize) - Common rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn (maize) - Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Corn (maize) - Healthy',
    'Grape___Black_rot': 'Grape - Black rot',
    'Grape___Esca_(Black_Measles)': 'Grape - Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape - Leaf blight (Isariopsis Leaf Spot)',
    'Grape___healthy': 'Grape - Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange - Haunglongbing (Citrus greening)',
    'Peach___Bacterial_spot': 'Peach - Bacterial spot',
    'Peach___healthy': 'Peach - Healthy',
    'Pepper,_bell___Bacterial_spot': 'Pepper, bell - Bacterial spot',
    'Pepper,_bell___healthy': 'Pepper, bell - Healthy',
    'Potato___Early_blight': 'Potato - Early blight',
    'Potato___Late_blight': 'Potato - Late blight',
    'Potato___healthy': 'Potato - Healthy',
    'Raspberry___healthy': 'Raspberry - Healthy',
    'Soybean___healthy': 'Soybean - Healthy',
    'Squash___Powdery_mildew': 'Squash - Powdery mildew',
    'Strawberry___Leaf_scorch': 'Strawberry - Leaf scorch',
    'Strawberry___healthy': 'Strawberry - Healthy',
    'Tomato___Bacterial_spot': 'Tomato - Bacterial spot',
    'Tomato___Early_blight': 'Tomato - Early blight',
    'Tomato___Late_blight': 'Tomato - Late blight',
    'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato - Septoria leaf spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato - Spider mites Two-spotted spider mite',
    'Tomato___Target_Spot': 'Tomato - Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato - Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato - Tomato mosaic virus',
    'Tomato___healthy': 'Tomato - Healthy'
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class = predict_image(filepath, model)
        predicted_class_label = class_labels[predicted_class]
        measures = precautionary_measures.get(predicted_class_label, [])
        pcm = class1_labels.get(predicted_class_label, [])

        return render_template('result.html', label=pcm, measures=measures)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
