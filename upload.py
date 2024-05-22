from flask import Flask, render_template, request, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img
from model import process_image, predict_class
import os

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)


app.config['UPLOADED_PHOTOS_DEST'] = './static/uploads'
configure_uploads(app, photos)


@app.route('/home', methods=['GET', 'POST'])
def home():
    welcome = "Hello, World !"
    return welcome


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        try:
            # Save the image
            filename = photos.save(request.files['photo'])
            print(f"Image saved as: {filename}")

            image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
            image = load_img(image_path, target_size=(224, 224))
            print("Image loaded")

            # Process the image
            image = process_image(image)
            print("Image processed")

            # Make prediction
            prediction, percentage = predict_class(image)
            print(f"Prediction: {prediction}, Probability: {percentage}")

            # Remove any '%' from the percentage string and convert to float
            percentage_float = float(percentage.strip('%')) / 100

            # Render the result template
            return render_template('result.html', prediction=prediction, percentage_float=percentage_float * 100, filename=filename)
        except Exception as e:
            print(f"Error: {e}")
            return f"<h1>Something went wrong: {e}</h1>"
   
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
