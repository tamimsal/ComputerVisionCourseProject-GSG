#MortyAI
#Computer Vision Course By Gaza Sky Geeks
#Done By: Tamim Salhab
# to run the program write python app.py on terminal 

#importing needed libraries
from flask import Flask, render_template, request
import numpy as np
import cv2
import os, shutil
from ultralytics import YOLO

#defining app using flask
app = Flask(__name__)                                                                       


#This section is for uploading the images
UPLOAD_FOLDER = 'static/uploads'                                                            
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Checking if file exists
if not os.path.exists(UPLOAD_FOLDER):                                                   
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

#Method to upload the images
@app.route('/', methods=['GET', 'POST'])
def UploadMultipleImagesFromLocalFolder():
    if request.method == 'POST':
        if 'image_uploads' not in request.files:
            return render_template('index.html', error='No file part')

    files = request.files.getlist('image_uploads')

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

    return render_template('index.html', success='Files uploaded successfully')

#Method to sort images by name
def SortingImagesByName(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

#Method to stritch the images using cv2 stritcher
def StrtichingImages():
    stitcher = cv2.Stitcher_create()
    files = SortingImagesByName('static/uploads')
    leftImage = cv2.imread('static/uploads/' + files[0])
    count = 0
    for file in files:
        count+=1
    
    if(count == 1):
        cv2.imwrite("static/results/result.jpg", leftImage)

    for file in files[1:]:
        rightImage = cv2.imread('static/uploads/' + file)

        status, stitched_image = stitcher.stitch((leftImage, rightImage))  
  
        if status == cv2.Stitcher_OK:
            print("Stitching successful!")
            leftImage = stitched_image
            cv2.imwrite("static/results/result.jpg", stitched_image)
            count+=1
        else:
            print("Stitching failed!")

#Method to human detection using YOLO
def HumansDetectionUsingYOLO():
    model = YOLO('yolov8n.pt') 
    results = model(['static/results/result.jpg'], classes = 0,conf = 0.5)  
    results[0].save(filename='static/results/resultHumanDetect.jpg') 

#Method to create diffrernce of guassian and enhanced DoG
def DifferenceOfGuassian(kernel_size):
    orginalImage = cv2.imread("static/results/result.jpg")
    grayImage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2GRAY)
    gaussian_1 = cv2.GaussianBlur(grayImage, (0, 0), 1)
    gaussian_3 = cv2.GaussianBlur(grayImage, (0, 0), 3)
    DoG = gaussian_1 - gaussian_3
    cv2.imwrite("static/results/DoG.jpg", DoG)

    enhanced_dog = cv2.morphologyEx(DoG, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
    cv2.imwrite("static/results/enhancedDoG.jpg", enhanced_dog)

#Method to create Canny edge detection method
def CannyEdgeDetection():
    orginalImage = cv2.imread("static/results/result.jpg")
    grayImage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2GRAY)
    median_value = np.median(grayImage)
    lower_threshold = int(max(0, 0.7 * median_value))
    upper_threshold = int(min(255, 1.3 * median_value))
    canny_edges = cv2.Canny(grayImage, lower_threshold, upper_threshold)
    cv2.imwrite("static/results/cannyResult.jpg", canny_edges)

#Method to run all the function when clicking a button
@app.route('/button_click', methods=['POST'])
def ButtonClick():
    if request.method == 'POST':
        path = "static/uploads"
        dir = os.listdir(path) 
        if len(dir) == 0: 
            print("Empty directory") 
            return render_template('index.html', success='Files uploaded successfully')
        else: 
            StrtichingImages()

            CannyEdgeDetection()
            HumansDetectionUsingYOLO()
            DifferenceOfGuassian(5)
            return render_template('hub.html', success='Files uploaded successfully')

#Method to go to strich page
@app.route('/stritch', methods=['POST', 'GET'])
def StritchPage():
    #if request.method == 'POST':
    imageList = os.listdir("static/uploads")
    imageList = ['uploads/' + image for image in imageList]
    return render_template("stritch.html", imageList = imageList)

#Method to go to human detection page
@app.route('/human', methods=['POST'])
def HumanDetectPage():
    if request.method == 'POST':
        return render_template('human.html', success='Files uploaded successfully')

#Method to go to edge detection page
@app.route('/EdgeDetect', methods=['POST'])
def EdgeDetectionPage():
    if request.method == 'POST':
        return render_template('edgeDetection.html', success='Files uploaded successfully')

#Method to return to home page
@app.route('/returnToHome', methods=['POST'])
def ReturnToHomePage():
    if request.method == 'POST':
        shutil.rmtree('static/results')
        os.makedirs('static/results')
        shutil.rmtree('static/uploads')
        os.makedirs('static/uploads')
        return render_template('index.html', success='Files uploaded successfully')

#method to run the slider and get values from it
@app.route("/slider11", methods=["POST"])
def GetSliderValue():
    name_of_slider = request.form["name_of_slider"]
    DifferenceOfGuassian(int(name_of_slider))
    return render_template('edgeDetection.html', success='Files uploaded successfully')

#Method to go back to main page
@app.route('/gettingBack', methods=['POST'])
def GettingBacktoMainPage():
    if request.method == 'POST':
        return render_template('hub.html', success='Files uploaded successfully')
    
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

