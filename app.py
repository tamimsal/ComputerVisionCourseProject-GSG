import os
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


@app.route('/', methods=['GET', 'POST'])
def index():
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

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)


