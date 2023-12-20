from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
from ml_with_wiki import BirdInfo
import os


UPLOAD_FOLDER = 'static\\user_uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def call_model(path):
    model = BirdInfo()
    model.get_input(path)
    pred = model.get_prediction()
    return model.get_wiki(pred)


@app.route("/")
def home():
    return render_template('h1.html', name='home')


@app.route("/info")
def home2():
    return render_template('h2.html', name='home2')


@app.route("/main", methods=['GET', 'POST'])
def home3():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Error!')
            return 'Error!'
        
        file = request.files['file']

        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = filename.replace(' ', '')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            data = call_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('h3.html', pic=filename, title=data['title'], summary=data['summary'], src=data['src'])

    return render_template('h3.html')


if __name__ == '__main__':
    app.run(port=8000)