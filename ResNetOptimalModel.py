"""
ResNet model for regression of Keras.
Optimal model for the paper

"Chen, D.; Hu, F.; Nian, G.; Yang, T. Deep Residual Learning for Nonlinear Regression. Entropy 2020, 22, 193."

Depth:28
Width:16
"""
from tensorflow.keras import layers,models
from tensorflow.keras import callbacks
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error


def identity_block(input_tensor,units):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		units:output shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	x = layers.add([x, input_tensor])
	x = layers.Activation('relu')(x)

	return x

def dens_block(input_tensor,units):
	"""A block that has a dense layer at shortcut.
	# Arguments
		input_tensor: input tensor
		unit: output tensor shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	shortcut = layers.Dense(units)(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.add([x, shortcut])
	x = layers.Activation('relu')(x)
	return x


def ResNet50Regression():
	"""Instantiates the ResNet50 architecture.
	# Arguments        
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as input for the model.        
	# Returns
		A Keras model instance.
	"""
	Res_input = layers.Input(shape=(7,))

	width = 16

	x = dens_block(Res_input,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)
	
	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = layers.BatchNormalization()(x)
	x = layers.Dense(1, activation='linear')(x)
	model = models.Model(inputs=Res_input, outputs=x)

	return model

#################################Prepare data####################################
plt.switch_backend('agg')
#path = "~/pub/dwchen/testData/min4008001200.csv"
path = "~/pub/dwchen/testData/min400800.csv"
dataSet = pd.read_csv(path)
dataSet = np.array(dataSet)

x = dataSet[:,0:7]
y = dataSet[:,7]
y = y.reshape(-1,1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x)
xscale = scaler_x.transform(x)
scaler_y.fit(y)
yscale = scaler_y.transform(y)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale,test_size=0.25)

##############################Build Model################################
model = ResNet50Regression()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()

#compute running time
starttime = datetime.datetime.now()

history = model.fit(X_train, y_train, epochs=50, batch_size=5000, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10,verbose=2, mode='auto')], validation_split=0.1)
#history = model.fit(X_train, y_train, epochs=10, batch_size=60000,  verbose=1, validation_split=0.1)
endtime = datetime.datetime.now()

##############################Save Model#################################
model.save('OptimalModelDataSet2.h5')
#plot_model(model, to_file='ResnetModel.png')
#from keras.models import load_model
#model.save('my_model.h5') 
#model = load_model('my_model.h5') 

#############################Model Predicting#################################
yhat = model.predict(X_test)

print('The time cost: ')
print(endtime - starttime)
print('The test loss: ')
print(mean_squared_error(yhat,y_test))

#invert normalize
yhat = scaler_y.inverse_transform(yhat) 
y_test = scaler_y.inverse_transform(y_test) 


###############################Visualize Model################################
# "Loss"
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()
plt.savefig('OptimalModelDataSet2.png')

plt.figure()
plt.plot(y_test[0:4000],'rx')
plt.plot(yhat[0:4000],' go',markerfacecolor='none')
plt.title('Result for ResNet Regression')
plt.ylabel('Y value')
plt.xlabel('Instance')
plt.legend(['Real value', 'Predicted Value'], loc='upper right')
plt.savefig('OptimalModelDataSet2.png')
#plt.show()

file = open('/data92/pub/dwchen/testData/dataset2.txt','r+')
file.write('predicted ' + 'observed ' + '\n')
for i in range(len(yhat)):
	file.write(str(yhat[i][0])+' '+str(y_test[i][0])+'\n')
file.close()



