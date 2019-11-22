# create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy


# fix random seed for reproducibility
'''seed = 7
numpy.random.seed(seed)'''


# load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset = numpy.loadtxt("ecoliData.csv", delimiter=",")

'''dataset=[
[1,2,2,1,0],
[5,2,2,5,0],
[1,2,2,3,1],
[7,3,3,8,1],
[1,7,7,2,2],
[1,4,4,3,2],
[1,1,2,3,3],
[2,2,5,1,3]
]
'''
# split into input (X) and output (Y) variables
X = dataset[:,:7]
Y = dataset[:,7:15]

'''print(X)
print(len(Y[0]))
exit(0)'''


# create model
model = Sequential()
model.add(Dense(8, input_dim=7, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='softmax'))


# compile model
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])


# input the dataset into created model
model.fit(X, Y, nb_epoch=150, batch_size=10)


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

'''for layer in model.layers:
    weights = layer.get_weights()[1] # list of numpy arrays
    print(weights)
    exit(0)'''

'''print("+++++++++++++++++++++++++++++++++++++")
print(model.get_weights())'''

#print(model.layers[1].get_weights()[0])

predictions = model.predict(X[0:1])
print(predictions[0])
