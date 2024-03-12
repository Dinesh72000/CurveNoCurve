from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)

dic = {0 : 'Curve', 1 : 'Not a Curve'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	img = image.load_img(img_path, target_size=(200,200))
	i = image.img_to_array(img)
	i=np.expand_dims(i,axis=0)
	images=np.vstack([i])
	p = model.predict(images)
	return dic[p[0][0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)