# Convolutional Neural Network for Prediction

# Importing the Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initializing the classifier
classifier = Sequential()
# Adding Convolution Layer
classifier.add(Convolution2D(filters=32, 
                             input_shape=(64,64,1),
                             kernel_size=(3,3),
                             activation='relu'))
# Adding Max Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Flattening
classifier.add(Flatten())
# Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preparing the train/test data and training the model
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/train_set',
        target_size=(64, 64),
        batch_size=5,
        color_mode = 'grayscale',
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=5,
        color_mode = 'grayscale',
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=400, # No. of Images in train set
        epochs=10,
        validation_data=test_set,
        validation_steps=60) # No. of Images in test set

# Saving the Model
from keras.models import model_from_json

# Serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
