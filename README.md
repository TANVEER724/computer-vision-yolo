Computer Vision Object Detection

A real-time Computer Vision Object Detection application built with Python, YOLO (Ultralytics), OpenCV, and Streamlit.

The application allows users to upload an image and automatically detect objects using a pretrained YOLO model. Detection results are displayed with bounding boxes, class labels, and confidence scores.

🚀 Features
Object detection using YOLO
Image upload through a Streamlit interface
Bounding boxes around detected objects
Object class labels
Confidence scores
Fast and simple web interface
Supports common image formats
Easy to run locally and deploy
🛠️ Technologies Used
Python
Ultralytics YOLO
OpenCV
Streamlit
NumPy
Pillow
📁 Project Structure
computer-vision-project/
│
├── app.py
├── requirements.txt
├── README.md
├── yolov8n.pt
│
└── assets/
    └── sample images
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🔍 How It Works
Upload an image through the Streamlit interface.
The YOLO model processes the image.
Objects are detected automatically.
Bounding boxes are generated around detected objects.
Class names and confidence scores are displayed.
The processed image is shown in the web application.
🎯 Use Cases

This project can be used as a foundation for:

People detection
Vehicle detection
Object monitoring
Security applications
Smart surveillance
Computer vision prototypes
AI-powered image analysis
👨‍💻 Author

Tanveer Hussain Tabish

AI Engineer | Machine Learning | Computer Vision | AI Automation

GitHub: TANVEER724

📄 License

This project is intended for educational, portfolio, and development purposes.
