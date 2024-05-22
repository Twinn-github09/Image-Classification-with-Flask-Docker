import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
import numpy as np

model = tf.keras.models.load_model('model.h5', compile=False)

def process_image(image):
  
    image = img_to_array(image)

  
    Img = []
    image_dl = img_to_array(image)
    Img.append(image_dl)
    Img = np.array(Img)
    return Img

def predict_class(image):
   
    yhat = model.predict(image)
   
    if(yhat[0][0]>yhat[0][1]):
        prob = yhat[0][0]
        prediction = 'cat'
    else:
        prob = yhat[0][1]
        prediction = 'dog'
    percentage = '%.2f%%' % (prob*100)

    return prediction, percentage

if __name__ == '__main__':
 
    image = load_img('10.jpg', target_size=(224, 224))
    image = process_image(image)
    prediction, percentage = predict_class(image)
    print(prediction, percentage)