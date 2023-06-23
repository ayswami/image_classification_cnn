#importing required libraries 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import matplotlib.pyplot as plt

#loading the data and preparing it for further application
base_dir = './Data'
train_dir = './Data/train'                      
test_dir = './Data/test'                           
train_c19_dir = './Data/train/COVID19'       
train_nml_dir = './Data/train/NORMAL'              
train_pn_dir = './Data/train/PNEUMONIA'      
test_c19_dir = './Data/test/COVID19'          
test_nml_dir = './Data/test/NORMAL'                
test_pn_dir = './Data/test/PNEUMONIA'


'''--------------------------------------------PART1---------------------------------------------'''
#classification using VGGNet16 model
x=ImageDataGenerator()
train_generator = x.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))
test_generator = x.flow_from_directory(test_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

base_model = VGG16(input_shape = (224, 224, 3),include_top = False,weights = 'imagenet')
for layer in base_model.layers:
  layer.trainable = False
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
vgg_hist = model.fit(train_generator, validation_data = test_generator, steps_per_epoch = 120, epochs = 10)


#classification using ResNet50 model
y = ImageDataGenerator()
train_generator = y.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))
test_generator = y.flow_from_directory(test_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

base_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")
for layer in base_model.layers:
  layer.trainable = False
base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(1, activation='sigmoid'))
base_model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['acc'])
resnet_hist = base_model.fit(train_generator, validation_data = test_generator, steps_per_epoch = 120, epochs = 10)


#plotting the accuracies of both the models
plt.plot(vgg_history.history['acc'])
plt.plot(resnet_history.history['acc'])
plt.title('Comparing accuracies of both the models')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['VGGNet16','ResNet50'],loc='upper left')
plt.show()


'''--------------------------------------------------PART2--------------------------------------------------'''
#classification using newly proposed model
model = Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(7))
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['acc'])
hist = model.fit(train_generator,epochs = 10,validation_data = test_generator, steps_per_epoch = 120)

#plotting the accuracy for proposed models
plt.plot(hist.history['acc'])
plt.title('Accuracy for proposed model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()