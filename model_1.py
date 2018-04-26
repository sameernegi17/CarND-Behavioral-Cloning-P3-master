import pandas as pd
import numpy as np
import cv2
import sklearn

X_train = []
y_train = []
correction = 0.15
data_path = './'
data_frame = pd.read_csv(data_path+'driving_log.csv',usecols=[0,1,2,3])

for index,row in data_frame.iterrows():
	steering=row[3]
	image_path = row[1].strip()
	image = cv2.imread(data_path+image_path)
	temp_measurement = steering
	if image != None:
		X_train.append(image)
		y_train.append(temp_measurement)
		image_flipped = np.fliplr(image)
		measurement_flipped = -temp_measurement
		X_train.append(image_flipped)
		y_train.append(measurement_flipped)
		
	image_path = row[0].strip()
	image = cv2.imread(data_path+image_path) 
	temp_measurement = steering + correction
	if image != None:
		X_train.append(image)
		y_train.append(temp_measurement)
		image_flipped = np.fliplr(image)
		measurement_flipped = -temp_measurement
		X_train.append(image_flipped)
		y_train.append(measurement_flipped)
	
	image_path = row[2].strip()
	image = cv2.imread(data_path+image_path)
	temp_measurement = steering - correction
	if image != None:
		X_train.append(image)
		y_train.append(temp_measurement)
		image_flipped = np.fliplr(image)
		measurement_flipped = -temp_measurement
		X_train.append(image_flipped)
		y_train.append(measurement_flipped)
	


from sklearn.model_selection import train_test_split
X_train,y_train = sklearn.utils.shuffle(X_train,y_train)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size=0.2)

	
def generator(X_train,y_train,batch_size):
	while(True):
		for offset in range(0,len(X_train),batch_size):
			temp_images = X_train[offset:offset+batch_size]
			temp_measurement = y_train[offset:offset+batch_size]
			training_samples = np.asarray(temp_images)
			training_measurement = np.asarray(temp_measurement)
			yield (training_samples,training_measurement)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda,Cropping2D


gen = generator(X_train,y_train,32)
valid_gen = generator(X_valid,y_valid,32)

#steer = np.array([], dtype=np.float32)
#for i im range(1782):
#	_, y = next(gen)
#	steer = np.concatenate(steer, y)
	
# use matplotlib to plot histogram

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit_generator(gen, samples_per_epoch= len(X_train), validation_data=valid_gen, nb_val_samples=len(X_valid), nb_epoch=5)

model.save('model.h5')