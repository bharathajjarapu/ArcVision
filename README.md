# ArcVision - Image Recognition API
Made by Bharath, Abhisehk, & Harsh.

## Login
<img src="/images/Login.png" width="850" alt="login_screenshot">

## Index
<img src="/images/Index.png" width="850" alt="login_screenshot">

## Input
<img src="/images/Moon Knight.jpg" width="850" alt="login_screenshot">

## Output
<img src="/images/output.jpg" width="850" alt="login_screenshot">

## Introduction

This is a Flask-based Image Recognition API that allows users to upload images and receive predictions or information about the recognized objects. This project is designed as a hackathon product, providing a simple and efficient way to perform image recognition tasks using pre-trained machine learning model which is finetuned for smaller devices i.e YOLO Lite.

## Features

- **User-friendly Interface :** The API provides a simple web interface where users can upload images easily.
- **Object Detection :** Utilizes YOLO Lite Self made (You Only Look Once) model for object detection in images.
- **Output Visualization :** The recognized objects are highlighted in the output image with bounding boxes and labels.

## Getting Started

We are using Python Anywhere to host the server but you can do it on your Personal Computer

### Prerequisites

Make sure you have a Decent PC or Server with following installed:

- Python
- Flask
- OpenCV

### Installation 

1. Clone the Repository.

```pyton
git clone https://github.com/bharathajjarapu/HackerzRec.git
```

2. Change Directory.
   
```pyton
cd ArcVision
```

3. Install the Requirements

```python
pip install flask opencv-contrib-python
```

4. Then run the python file in terminal
```python
python app.py
```

5. Or If you are using PythonAnywhere run the python file in terminal
```python
python main.py
```

## Usage

1. Open your web browser and go to ```http://127.0.0.1:5000/```.
2. Upload an image using the provided form.
3. Click on the "Detect Object" button.
4. View the results on the result page, highlighting the recognized objects in the output image.

## Project Structure

- app.py: The main Flask application file containing the routes and image processing logic.
- templates/: Folder containing HTML templates for the web interface.
-  templates/index.html: Main page for image upload.
-  templates/result.html: Page displaying input and output images with recognized objects.

Happy hacking!
