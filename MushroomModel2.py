import PySimpleGUI as gui
from PIL import Image
import io
import os
import hashlib #used to in scrapeDatabse to detect duplicate images. 
import json #used to save and load in model information 

# Main libraries actually doing the ML stuff.
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Add in more metric data points 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)

#Helpers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

'''
#Constant, directory to the data itself. 
DATA_DIR=Path(__file__).resolve().parent/"assets"/"data"
RANDOM_STATE=42

#Data Saving items for Model saving and evaluation resutls 
MODEL_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "evaluation"

# Create directories if they do not already exist
MODEL_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)

#Constants. Images are resized to make processing faster, and because MobileNetV2 often uses 224x224. 
IMAGE_SIZE=(224,224)
BATCH_SIZE=4

# Maximum number of training epochs to keep computer running happy
EPOCHS = 15
'''
# Changed path ways for the project 
# Main directory where MushroomModel2.py is located
PROJECT_DIR = Path(__file__).resolve().parent

# Directory containing mushroom images
DATA_DIR = PROJECT_DIR / "assets" / "data"

# Directory for saving trained models
MODEL_DIR = PROJECT_DIR / "models"

# Directory for saving model evaluation results
EVALUATION_DIR = PROJECT_DIR / "evaluation"

# Create directories if they do not already exist
MODEL_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# MobileNetV2 image dimensions
IMAGE_SIZE = (224, 224)

# Number of images processed in each batch
BATCH_SIZE = 4

# Maximum number of training epochs
EPOCHS = 2

# Dataset split percentages
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Optional image augmentation
USE_AUGMENTATION = False

# Set random seeds
tf.keras.utils.set_random_seed(RANDOM_STATE)


imagePaths=[]
imageLabels=[]
model = None
#variables for training/testing data
testDataset= None
trainDataset = None
trainPaths=None
testPaths=None
trainLabels=None
testLabels=None

#Create GUI to import an image
def screen():
    image_path = None
    #Added in Validation Variables 
    model = None
    history = None
    testDataset = None
    evaluationResults = None

    # Create a blank image for resetting the image preview ensure vaild images are being shown 
    blankImage = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    blankBuffer = io.BytesIO()
    blankImage.save(blankBuffer, format="PNG")
    blankImageData = blankBuffer.getvalue()


    '''
    column = [
        [
            gui.Text("Setting up the model...\nThis could take a minute.", key="-SETUP_TEXT-")
        ],
        [
            gui.Text("Image to scan: ", key="-INPUT_TEXT-", visible=False),
            gui.Input(enable_events=True, key="-IMAGE_INPUT-", visible=False),
            gui.FileBrowse(file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.gif"),),key="-FILE_BROWSE-",visible=False),
        ],
        [
            gui.Image(size=(80,80), key="-IMAGE-"),
        ],
        [
            gui.Text("",key="-IMAGE_TEXT-"),
        ],
        [
            gui.Button("SCAN IMAGE", key="-SCAN_BUTTON-", visible=False),
            gui.Button("TEST MODEL", key="-TEST_MODEL_BUTTON-", visible = False),
            gui.Button("CLEAR IMAGE", key="-CLEAR_IMAGE-", visible = False)
        ],
        #Added in 
        [
            gui.Button("EVALUATE MODEL", key="-EVALUATE_MODEL-", visible = False),
            gui.Button("CONFUSION MATRIX", key="-CONFUSION_MATRIX-", visible=False)
        ],
        [
           gui.Button("TRAINING GRAPH", key="-TRAINING_GRAPHS-", visible=False),
           gui.Button("EXIT",key="-EXIT-")
        ],
        [
            gui.Multiline("", size=(65, 18), key="-TEST_RESULT_TEXT-", disabled=True)
        ],
        [
            gui.Multiline("Upload an image to begin.", size=(55, 12), key="-IMAGE_TEXT-", disabled=True)
        ],
        [
            gui.Text("", key="-TEST_RESULT_TEXT-")
        ]
    ]
    '''
    #Fixed Layout Section 
    column = [
        [
            gui.Text("Setting up the model...\nThis could take a minute.",key="-SETUP_TEXT-")
        ],
        [
            gui.Text("Image to scan: ",key="-INPUT_TEXT-",visible=False),
            gui.Input(enable_events=True,key="-IMAGE_INPUT-",visible=False),
            gui.FileBrowse(
                file_types=(
                ("Image Files", "*.png;*.jpg;*.jpeg;*.gif"),),
                target="-IMAGE_INPUT-",key="-FILE_BROWSE-",visible=False)
        ],
        [
            gui.Image(size=(80, 80), key="-IMAGE-")
        ],
        [
            # Display image classification results
            gui.Multiline("Upload an image to begin.",size=(55, 12), key="-IMAGE_TEXT-",disabled=True)
        ],
        [
            gui.Button( "SCAN IMAGE", key="-SCAN_BUTTON-", visible=False),
            gui.Button("TEST MODEL",key="-TEST_MODEL_BUTTON-",visible=False),
            gui.Button("CLEAR IMAGE", key="-CLEAR_IMAGE-",visible=False)
        ],
        [
            #Vaildation buttons added in 
            gui.Button("EVALUATE MODEL", key="-EVALUATE_MODEL-",visible=False),
            gui.Button("CONFUSION MATRIX", key="-CONFUSION_MATRIX-",visible=False)
        ],
        [
            gui.Button("TRAINING GRAPHS",key="-TRAINING_GRAPHS-",visible=False),
            gui.Button("EXIT", key="-EXIT-")
        ],
        [
            # Display overall model evaluation results
            gui.Multiline("",size=(65, 18),key="-TEST_RESULT_TEXT-",disabled=True)
        ]
    ]


    layout = [
        [gui.Column(column, key="-IMAGE_COLUMN-")]
    ]
    window = gui.Window("Mushroom Identifier", layout, finalize=True, resizable=True)
    window.refresh()

        
    #Set up the model
    print("WORKING ON SETUP")

    # Collect mushroom images and species names
    classNames = scrapeDatabase()

    # Split the dataset into three parts
    (trainPaths, valPaths, testPaths, trainLabels, valLabels, testLabels) = splitDatabase(classNames)

    # Create the training dataset
    trainDataset = makeDataset(
        trainPaths,
        trainLabels,
        augment=USE_AUGMENTATION,
        shuffle=True
    )

    # Create the validation dataset
    valDataset = makeDataset(
        valPaths,
        valLabels,
        augment=False,
        shuffle=False
    )

    # Create the testing dataset
    testDataset = makeDataset(
        testPaths,
        testLabels,
        augment=False,
        shuffle=False
    )

    # Update the GUI during model training
    window["-SETUP_TEXT-"].update("Training MobileNetV2...\n" "See VS Code terminal for progress.")

    window.refresh()

    # Train the model and store training history
    model, history = makeModel(
        trainDataset,
        classNames,
        valDataset
    )

    # Save the trained model
    saveModel(
        model,
        history,
        classNames,
        testPaths,
        testLabels
    )

    # Store evaluation results when available
    evaluationResults = None



    #Open up the UI to be used
    window["-SETUP_TEXT-"].update(visible=False)
    window["-INPUT_TEXT-"].update(visible=True)
    window["-IMAGE_INPUT-"].update(visible=True)
    window["-TEST_MODEL_BUTTON-"].update(visible=True)
    window["-FILE_BROWSE-"].update(visible=True)
    window["-SCAN_BUTTON-"].update(visible=True) #Change was made here 
    
    # Make the new evaluation buttons available for validation 
    window["-CLEAR_IMAGE-"].update(visible=True)
    window["-EVALUATE_MODEL-"].update(visible=True)
    window["-CONFUSION_MATRIX-"].update(visible=True)
    window["-TRAINING_GRAPHS-"].update(visible=True)

    #GUI loop to render window
    while True:
        event, values = window.read()

        '''
        if event=="-TEST_MODEL_BUTTON-":
            window["-TEST_RESULT_TEXT-"].update("Testing the model...")
            testLoss,testAccuracy=model.evaluate(testDataset)
            window["-TEST_RESULT_TEXT-"].update("Test Loss: "+ str(testLoss) + "\n"+"Test Accuracy: "+str(testAccuracy))
        '''

        '''
        if event=="-IMAGE_INPUT-":
            filename = values["-IMAGE_INPUT-"]
            image = Image.open(filename)
            image.thumbnail((400, 400))  # Caps max dimensions to 400x400
            image_path = values["-IMAGE_INPUT-"]

            # Convert image so it can be displayed
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            # Display the image, show SCAN btn
            window["-IMAGE-"].update(data=bio.getvalue())
            window["-IMAGE_TEXT-"].update("")
            window["-SCAN_BUTTON-"].update(visible=True)
        '''

        '''
        if event=="-SCAN_BUTTON-" and image_path:
            #add in if no image is selected and display an error message here to be added
            window["-SCAN_BUTTON-"].update(visible=False)
            window["-IMAGE_TEXT-"].update("Running image through the model...")
            converted_image = loadImage(image_path)[0]
            #Make the converted image look like a batch of images so the model doesn't fuss
            prediction = model.predict(tf.expand_dims(converted_image, axis=0))[0]
            index = int(np.argmax(prediction))
            name = classNames[index]
            confidence = float(prediction[index])
            window["-IMAGE_TEXT-"].update("Predicted class: "+name+"\n"+"Probability: "+str(confidence))
        '''
        #Close the program 
        if event == gui.WIN_CLOSED or event == "-EXIT-":
            break
        
        # UPLOAD AN IMAGE
        elif event == "-IMAGE_INPUT-":

            filename = values["-IMAGE_INPUT-"]

            # Reset previous image selection
            image_path = None

            window["-IMAGE-"].update(data=blankImageData)
            window["-IMAGE_TEXT-"].update("")

            # Check whether an image was selected
            if not filename:
                window["-IMAGE_TEXT-"].update("No image has been selected.")
                continue

            # Check whether the file exists
            if not os.path.isfile(filename):
                window["-IMAGE_TEXT-"].update("The selected image could not be found.")
                continue

            try:
                # Open and prepare the image for display
                with Image.open(filename) as originalImage:

                    image = originalImage.convert("RGB")
                    image.thumbnail((400, 400))

                    # Convert image to PNG bytes
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")

                # Confirm TensorFlow can process the image
                loadImage(filename)

                # Display the image
                window["-IMAGE-"].update(data=bio.getvalue())

                # Save the selected image path
                image_path = filename

                window["-IMAGE_TEXT-"].update("Image uploaded successfully.\n""Ready to scan.")

            except Exception as error:
                image_path = None
                gui.popup_error( "Unable to open the selected image.",str(error))
                window["-IMAGE_TEXT-"].update("Please select a valid image.")

        # SCAN IMAGE OR TEST MODEL
        elif event in (
            "-SCAN_BUTTON-",
            "-TEST_MODEL_BUTTON-"
        ):
            # Check whether an image is available
            if not image_path:
                if event == "-TEST_MODEL_BUTTON-":
                    message = ("Test model does not have ""an image available to test.")

                else:
                    message = ("No image is available to scan.")

                gui.popup_error(message,"Please upload an image first.")

                window["-IMAGE_TEXT-"].update(message)
                continue

            # Check if the selected image still exists
            if not os.path.isfile(image_path):

                image_path = None

                window["-IMAGE-"].update(data=blankImageData)

                gui.popup_error("The selected image is no longer available.")

                window["-IMAGE_TEXT-"].update("Please upload another image.")
                continue

            try:
                window["-IMAGE_TEXT-"].update("Running image through the model...")

                window.refresh()

                # Call the prediction function
                resultsText = predictMushroom(model,image_path,classNames)

                # Display and retain prediction results
                window["-IMAGE_TEXT-"].update(resultsText)

                # Do not hide the scan button
                # Do not remove the uploaded image

            except Exception as error:
                gui.popup_error("Image classification failed.",str(error))

        # CLEAR IMAGE
        elif event == "-CLEAR_IMAGE-":
            image_path = None
            window["-IMAGE_INPUT-"].update("")
            window["-IMAGE-"].update(data=blankImageData)
            window["-IMAGE_TEXT-"].update("Upload an image to begin.")

        
        # EVALUATE MODEL
        elif event == "-EVALUATE_MODEL-":
            try:
                window["-TEST_RESULT_TEXT-"].update("Evaluating the model...")

                window.refresh()

                # Calculate all model evaluation metrics
                evaluationResults = evaluateModel(model, testDataset, classNames)

                # Display evaluation results in the GUI
                window["-TEST_RESULT_TEXT-"].update(evaluationResults["text"])

            except Exception as error:
                evaluationResults = None

                gui.popup_error("Model evaluation failed.", str(error))

                window["-TEST_RESULT_TEXT-"].update("Model evaluation failed.")

        
        # CONFUSION MATRIX
        elif event == "-CONFUSION_MATRIX-":
            # Evaluation must be completed first
            if evaluationResults is None:
                gui.popup_error("No evaluation results are available.", "Please click EVALUATE MODEL first.")
                continue

            try:
                # Display the confusion matrix
                displayConfusionMatrix(evaluationResults["matrix"],classNames)

            except Exception as error:
                gui.popup_error("Unable to display confusion matrix.", str(error))

        
        # TRAINING GRAPHS
        elif event == "-TRAINING_GRAPHS-":
            if history is None:
                gui.popup_error("No training history is available.")
                continue

            try:
                # Display training and validation graphs
                displayTrainingGraphs(history)

            except Exception as error:

                gui.popup_error(
                    "Unable to display training graphs.",
                    str(error)
                )

    # Close the GUI window
    window.close()

'''
def scrapeDatabase():
    
    #Goes through each directory in the folder and makes a list out of them. 
    classNames=sorted(
        directory.name
        for directory in DATA_DIR.iterdir()
        if directory.is_dir()
    )
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
    return classNames
'''

#Updated Version of function 
# Helps to check if images are vaild and find duplicates and takes action. 
# Helps to clean data collect and prepocessing for using good data. 
def scrapeDatabase():

    # Clear previous data to prevent duplicate entries
    imagePaths.clear()
    imageLabels.clear()

    # Check whether the dataset folder exists
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder was not found: {DATA_DIR}")

    # Get mushroom species from folder names
    classNames = sorted(
        directory.name
        for directory in DATA_DIR.iterdir()
        if directory.is_dir()
    )

    if len(classNames) < 2:
        raise ValueError("At least two mushroom species are required.")

    # Supported image file formats
    supportedFormats = {".png", ".jpg", ".jpeg", ".gif"}

    # Used to identify exact duplicate files Check for duplicates 
    seenImages = set()

    duplicateCount = 0
    invalidCount = 0

    # Go through every mushroom species folder
    for classIndex, className in enumerate(classNames):

        classDirectory = DATA_DIR / className
        classCount = 0

        for imagePath in sorted(classDirectory.iterdir()):

            # Ignore unsupported files
            if not imagePath.is_file():
                continue

            if imagePath.suffix.lower() not in supportedFormats:
                continue

            try:
                # Check whether the image can be opened
                with Image.open(imagePath) as image:
                    image.verify()

                # Confirm that TensorFlow can decode it
                loadImage(str(imagePath), classIndex)

                # Create a unique hash of the file contents
                imageHash = hashlib.sha256(
                    imagePath.read_bytes()
                ).hexdigest()

                # Skip exact duplicate images
                if imageHash in seenImages:
                    duplicateCount += 1
                    print("Duplicate skipped:", imagePath)
                    continue

                seenImages.add(imageHash)

                # Add valid image and corresponding label
                imagePaths.append(str(imagePath))
                imageLabels.append(classIndex)
                classCount += 1

            except Exception as error:
                invalidCount += 1
                print("Invalid image skipped:", imagePath)
                print("Reason:", error)

        # Display number of valid images for each species
        print(f"{className}: {classCount} valid images")

        # Each species needs at least three images
        if classCount < 3:

            raise ValueError(
                f"{className} has only {classCount} valid images. "
                "At least three are required for training, "
                "validation, and testing."
            )

        # Warn about species with very little data
        if classCount < 20:

            print(
                f"WARNING: {className} has limited data. "
                "Evaluation results may be unreliable."
            )

    print("\nDATASET SUMMARY")
    print("Image Count:", len(imagePaths))
    print("Class Count:", len(classNames))
    print("Duplicates Removed:", duplicateCount)
    print("Invalid Images Removed:", invalidCount)

    return classNames


# Split the mushroom dataset into training, validation and testing
def splitDatabase(classNames):

    # First split:
    # 70% training and 30% temporary data
    trainPaths, tempPaths, trainLabels, tempLabels = train_test_split(
        imagePaths,
        imageLabels,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=imageLabels
    )

    # Second split:
    # Divide the remaining 30% into validation and testing
    valPaths, testPaths, valLabels, testLabels = train_test_split(
        tempPaths,
        tempLabels,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=tempLabels
    )

    # Make sure images do not overlap between datasets
    assert set(trainPaths).isdisjoint(valPaths)
    assert set(trainPaths).isdisjoint(testPaths)
    assert set(valPaths).isdisjoint(testPaths)

    # Display dataset information
    print("\nDATASET SPLIT")
    print("----------------------------------------")
    print("Training Count:", len(trainPaths))
    print("Validation Count:", len(valPaths))
    print("Testing Count:", len(testPaths))

    # Display image counts for each mushroom species
    print("\nIMAGES PER SPECIES")

    for classIndex, className in enumerate(classNames):

        trainCount = trainLabels.count(classIndex)
        valCount = valLabels.count(classIndex)
        testCount = testLabels.count(classIndex)

        print(
            f"{className}: "
            f"Training={trainCount}, "
            f"Validation={valCount}, "
            f"Testing={testCount}"
        )

    # Return the paths and labels for all three datasets
    return (
        trainPaths,
        valPaths,
        testPaths,
        trainLabels,
        valLabels,
        testLabels
    )


#Loading in the image and making the size of the image readable/consisitent 
def loadImage (path, label=None):
        image = tf.io.read_file(path)

        # Channels for RGB, expand_animations is only there in case an animated file winds up in the set somehow.
        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        #Specify the expected image shape 
        image.set_shape([None, None, 3])

        image = tf.image.resize(image,IMAGE_SIZE)

        #Normalizes pixels from 0-255 to [-1,1]. 
        image=tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label



def makeDataset(paths, labels, augment=False, shuffle=False):
    
    #Prevent a empty dataset from being created 
    if len(paths) == 0:
        raise ValueError ("Cannot create a dataset without images")

    # Convert paths and labels into a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset=dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
            )


    #Loads the images into the dataset in parallel (preprocess images)
    dataset=dataset.map(loadImage,num_parallel_calls=tf.data.AUTOTUNE)

    #Alters the images a little to help prevent overfitting. Optional. 
    if augment: 
        augmentation=tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ])

        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True),label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    #Group the images into batches and prepare for upcoming data 
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def makeModel(trainDataset, classNames, valDataset):
    #Load in Pretrained MoblieNetV2 Model 
    baseModel=tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet",
    )

    #Freeze the pertrained layers 
    baseModel.trainable=False

    #Create the mushroom classification model 
    model = tf.keras.Sequential([
        baseModel,
        tf.keras.layers.GlobalAveragePooling2D(), #reduce extracted image features
        tf.keras.layers.Dropout(0.25), #prevents overfitting 
        #one output for each mushroom species 
        tf.keras.layers.Dense(len(classNames), activation="softmax"),
    ])

    #How the model learns 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    #Vaildation of the Model 
     # Stop training when validation loss stops improving
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # Train the model using training and validation data
    history = model.fit(
        trainDataset,
        validation_data=valDataset,
        epochs=EPOCHS,
        callbacks=[earlyStopping],
        verbose=1
    )

    #This old model was being used for one epoch where its train count was specified
    #model.fit(trainDataset, shuffle=False)

    #return model

    # Return the trained model and training history
    return model, history

#Adding in Validation Metrics systmes 
def evaluateModel(model, testDataset, classNames):

    # Store actual and predicted mushroom labels
    actualLabels = []
    predictedLabels = []

    # Calculate test loss and accuracy
    testLoss, testAccuracy = model.evaluate(testDataset, verbose=0)

    # Go through each batch of testing images
    for images, labels in testDataset:

        # Predict the species of each image
        predictions = model.predict(images,verbose=0)

        # Get the class with the highest probability
        predictedClasses = np.argmax(predictions, axis=1)

        # Store actual and predicted classes
        actualLabels.extend(labels.numpy())
        predictedLabels.extend(predictedClasses)

    # Convert lists into NumPy arrays
    actualLabels = np.array(actualLabels)
    predictedLabels = np.array(predictedLabels)

    # Include every mushroom species
    classIDs = list(range(len(classNames)))

    # Calculate accuracy (overall model performance)
    accuracy = accuracy_score(
        actualLabels,
        predictedLabels
    )

    # Calculate macro precision (How reliable a species prediction is)
    precision = precision_score(
        actualLabels,
        predictedLabels,
        labels=classIDs,
        average="macro",
        zero_division=0
    )

    # Calculate macro recall (How well model identifes a particular species)
    recall = recall_score(
        actualLabels,
        predictedLabels,
        labels=classIDs,
        average="macro",
        zero_division=0
    )

    # Calculate macro F1 score (balanced classifcation perfromance)
    f1 = f1_score(
        actualLabels,
        predictedLabels,
        labels=classIDs,
        average="macro",
        zero_division=0
    )

    # Generate a report for each mushroom species
    report = classification_report(
        actualLabels,
        predictedLabels,
        labels=classIDs,
        target_names=classNames,
        zero_division=0
    )

    # Generate the confusion matrix
    matrix = confusion_matrix(actualLabels, predictedLabels, labels=classIDs)

    # Format the evaluation results helps to make it readable
    resultsText = (
        "MOBILENETV2 MODEL EVALUATION\n"
        "====================================\n\n"

        f"Testing Images: {len(actualLabels)}\n"
        f"Test Loss: {testLoss:.4f}\n"
        f"Test Accuracy: {accuracy:.2%}\n"
        f"Macro Precision: {precision:.2%}\n"
        f"Macro Recall: {recall:.2%}\n"
        f"Macro F1 Score: {f1:.2%}\n\n"

        "CLASSIFICATION REPORT\n"
        "====================================\n\n"

        f"{report}"
    )

    # Display the results in the VS Code terminal
    print(resultsText)

    # Save the results for the final report
    with open(
        EVALUATION_DIR / "model_results.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(resultsText)

    # Return evaluation information in summary list 
    return {
        "loss": float(testLoss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report": report,
        "matrix": matrix,
        "text": resultsText
    }

#Add in the condfusion Matrix 
# IN reprot talk about the strength and weaknesss of the model 
def displayConfusionMatrix(matrix, classNames):

    # Create the graph
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create confusion matrix visualization
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classNames)

    # Plot the confusion matrix
    display.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format="d")

    # Add graph labels
    ax.set_title("MobileNetV2 Mushroom Confusion Matrix")
    ax.set_xlabel("Predicted Mushroom Species")
    ax.set_ylabel("Actual Mushroom Species")

    # Prevent labels from being cut off
    fig.tight_layout()

    # Save the graph
    fig.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Display the graph
    plt.show()

    # Close the figure after displaying it
    plt.close(fig)

#Output the graphs based on the modle training 
def displayTrainingGraphs(history):

    # Support both new and previously saved history
    if hasattr(history, "history"):
        historyData = history.history
    else:
        historyData = history

    # Determine how many epochs were completed
    epochs = range(1, len(historyData["accuracy"]) + 1)

    
    # ACCURACY GRAPH
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, historyData["accuracy"], label="Training Accuracy")

    ax1.plot(epochs, historyData["val_accuracy"], label="Validation Accuracy")

    ax1.set_title("MobileNetV2 Training and Validation Accuracy")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    ax1.legend()
    ax1.grid(True)

    fig1.tight_layout()

    # Save accuracy graph
    fig1.savefig(EVALUATION_DIR / "model_accuracy.png", dpi=300, bbox_inches="tight")

    # LOSS GRAPH
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(epochs, historyData["loss"], label="Training Loss")

    ax2.plot(epochs, historyData["val_loss"], label="Validation Loss")

    ax2.set_title("MobileNetV2 Training and Validation Loss")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    ax2.legend()
    ax2.grid(True)

    fig2.tight_layout()

    # Save loss graph
    fig2.savefig(
        EVALUATION_DIR / "model_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Display both graphs
    plt.show()

    plt.close(fig1)
    plt.close(fig2)


#Individual mushroom prediction 
def predictMushroom(model, imagePath, classNames):

    # Load and preprocess the selected image
    convertedImage = loadImage(imagePath)[0]

    # Add a batch dimension for MobileNetV2
    convertedImage = tf.expand_dims(convertedImage, axis=0)

    # Generate prediction probabilities
    prediction = model.predict(convertedImage, verbose=0)[0]

    # Get the three most likely species
    topIndices = np.argsort(
        prediction
    )[::-1][:3]

    # Create the results message
    resultsText = (
        "MUSHROOM CLASSIFICATION RESULTS\n"
        "================================\n\n"
    )

    # Display the top three predictions
    for rank, index in enumerate(topIndices,start=1):
        name = classNames[index]

        confidence = float(prediction[index] )

        resultsText += (
            f"{rank}. {name}\n"
            f"Confidence: {confidence:.2%}\n\n"
        )

    # Add a safety disclaimer
    resultsText += (
        "Text to be added to mushroom resutls fix this soon"
    )
    return resultsText


#Model saving and loading 
#This will save the trained model 
def saveModel(
        model,
        history,
        classNames,
        testPaths,
        testLabels):

    # Save the trained MobileNetV2 model
    model.save(MODEL_DIR / "mushroom_model.keras")

    # Save species names in the correct order
    with open(
        MODEL_DIR / "class_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(classNames,file,indent=4)

    # Convert training history into JSON-compatible data
    historyData = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    # Save training history
    with open(
        MODEL_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(historyData,file,indent=4)

    # Preserve the original testing dataset
    # Relative paths make the project more portable
    testData = {
        "paths": [
            str(Path(path).relative_to(PROJECT_DIR))
            for path in testPaths
        ],
        "labels": [
            int(label)
            for label in testLabels
        ]
    }

    with open(
        MODEL_DIR / "test_split.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(testData,file,indent=4)

    print("Model saved successfully.")

#Load in an existing model 
# Good to have as don't want to load in trained model and evaluate it using images its trained on 
def loadSavedModel():
    # Load the trained model
    model = tf.keras.models.load_model(
        MODEL_DIR / "mushroom_model.keras"
    )

    # Load mushroom species names
    with open(
        MODEL_DIR / "class_names.json",
        "r",
        encoding="utf-8"
    ) as file:
        classNames = json.load(file)

    # Load previous training history
    with open(
        MODEL_DIR / "training_history.json",
        "r",
        encoding="utf-8"
    ) as file:
        history = json.load(file)

    # Load the original testing split
    with open(
        MODEL_DIR / "test_split.json",
        "r",
        encoding="utf-8"
    ) as file:
        testData = json.load(file)

    # Reconstruct the testing image paths
    testPaths = [

        str(PROJECT_DIR / path)
        for path in testData["paths"]
    ]

    testLabels = testData["labels"]

    # Verify that the testing files still exist
    for path in testPaths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Testing image is missing: {path}"
            )

    # Reconstruct the original testing dataset
    testDataset = makeDataset( testPaths, testLabels)

    print("Saved model loaded successfully.")

    return (model, history, classNames, testDataset)

#Make it appear on the screen 
screen()