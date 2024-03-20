from flask import Flask, render_template, request, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def sorted_directory_listing_with_os_listdir(directory):
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items

def stritchImages():
    stitcher = cv2.Stitcher_create()
    files = sorted_directory_listing_with_os_listdir('uploads')
    leftImage = cv2.imread('uploads/' + files[0])
    count = 0
    for file in files[1:]:
        rightImage = cv2.imread('uploads/' + file)
        status, stitched_image = stitcher.stitch((leftImage, rightImage))  
  
        if status == cv2.Stitcher_OK:
            print("Stitching successful!")
            leftImage = stitched_image
            cv2.imwrite("uploads/result.jpg", stitched_image)
            count+=1
        else:
            print("Stitching failed!")




def cannyEdgeDetection():
    orginalImage = cv2.imread("uploads/result.jpg")
    grayImage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2GRAY)
    median_value = np.median(grayImage)
    lower_threshold = int(max(0, 0.7 * median_value))
    upper_threshold = int(min(255, 1.3 * median_value))
    canny_edges = cv2.Canny(grayImage, lower_threshold, upper_threshold)
    cv2.imwrite("uploads/cannyResult.jpg", canny_edges)

@app.route('/', methods=['GET', 'POST'])
def uploadImages():
    if request.method == 'POST':
    # Check if the post request has the file part
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
    




@app.route('/button_click', methods=['POST'])
def button_click():
    if request.method == 'POST':
        stritchImages()
        cannyEdgeDetection()
        return render_template('index.html', success='Files uploaded successfully')


@app.route('/stritch', methods=['POST'])
def stritch():
    if request.method == 'POST':
        return render_template('stritch.html', success='Files uploaded successfully')








def index():
    uploadImages()
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


