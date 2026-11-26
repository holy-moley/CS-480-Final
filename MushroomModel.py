# Main libraries actually doing the ML stuff.
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Helpers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#Constant, directory to the data itself. 
DATA_DIR=Path(__file__).resolve().parent/"assets"/"data"
RANDOM_STATE=42

#Goes through each directory in the folder and makes a list out of them. 
classNames=sorted(
    directory.name
    for directory in DATA_DIR.iterdir()
    if directory.is_dir()
)

imagePaths=[]
imageLabels=[]

#Nested for loop, looks through each class, gets all the file paths lined out. 
#This could almost definitely be made way more efficient. 
for classIndex, className in enumerate(classNames):
    classDirectory=DATA_DIR / className

    for imagePath in classDirectory.iterdir():
        #Only actually includes pngs, but I figure it's best to allow multiple options, in case additional files are added. 
        if imagePath.suffix.lower()in{".png",".jpg",".jpeg",".bmp"}:
            imagePaths.append(str(imagePath))
            imageLabels.append(classIndex)

print("Image Count: ", len(imagePaths))
print("Class Count: ", len(classNames))

#I would like to do a validation set as well, but we only have four images of truffles.
# 75/25 train/test split.  
#Important: stratifies by category. 

trainPaths, testPaths, trainLabels, testLabels = train_test_split(
    imagePaths, 
    imageLabels, 
    test_size=0.20, 
    random_state=RANDOM_STATE,
    stratify=imageLabels,
    )

#Makes extra sure that the training and testing paths don't overlap. This shouldn't be relevant unless scikit messes up, though.

assert set (trainPaths).isdisjoint(testPaths)

print("\nTraining count: ", len(trainPaths))
print("Test count: ", len(testPaths))

#Constants. Images are resized to make processing faster, and because MobileNetV2 often uses 224x224. 
IMAGE_SIZE=(224,224)
BATCH_SIZE=4

def makeDataset(paths, labels, augment=False, shuffle=False):
    dataset=tf.data.Dataset.from_tensor_slices((paths,labels))
    if shuffle:
        dataset=dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
            )

    def loadImage (path, label):
        image = tf.io.read_file(path)
        # Channels for RGB, expand_animations is only there in case an animated file winds up in the set somehow.
        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )
        image = tf.image.resize(image,IMAGE_SIZE)
        #Normalizes pixels from 0-255 to [-1,1]. 
        image=tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    dataset=dataset.map(loadImage,num_parallel_calls=tf.data.AUTOTUNE)

    #Alters the images a little to help prevent overfitting. Optional. 
    if augment: 
        augmentation=tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ])
        dataset=dataset.map(
            lambda image, label: (augmentation(image),label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#Augments in initial testing actually reduced performance, but they will be kept as an option for future development. 
trainDataset=makeDataset(trainPaths,trainLabels,augment=True, shuffle=True)
testDataset=makeDataset(testPaths,testLabels)

baseModel=tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet",
)
baseModel.trainable=False

model = tf.keras.Sequential([
    baseModel,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(len(classNames), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(trainDataset, shuffle=False)
testLoss,testAccuracy=model.evaluate(testDataset)
print("\n Test accuracy: ", testAccuracy)