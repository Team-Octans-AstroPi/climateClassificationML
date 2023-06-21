from pathlib import Path
import numpy as np

# Edge TPU
from PIL import Image                                   # resize images
from pycoral.adapters import common, classify           # used by Coral Edge TPU
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from tflite_runtime import interpreter


script_dir = Path(__file__).parent.resolve()
model_file = script_dir/'climate-model.tflite'
label_file = script_dir/'labels.txt'

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

def analyseDataset():
    # Pandas - dataframe
    import pandas as pd

    # Load the CSV file into a pandas dataframe, while specifying the column names
    data = pd.read_csv('~/astropi/final-dataset.csv', names=["imagename", "temperature", "precipitationIntensity", "vegetationPercent", "latitude", "longitude", "plantHealth1", "plantHealth2", "plantHealth3", "plantHealth4", "plantHealth5", "plantHealth6", "plantHealth7", "plantHealth8", "climateID", "cloudType", "landType"])

    # Separate features and labels, while removing unnecessary columns
    features = data.drop('longitude', axis=1).drop('imagename', axis=1).drop('climateID', axis=1)
    labels = data['climateID']



    # Replace the cloud type strings in the dataframe with integer ids,
    # by using the map which was used when the model was compiled.
    
    # Make sure to replace the map with your model's one.
    cloud_map = {'cirrocumulus': 0, 'none': 1, 'nimbostratus': 2, 'cumulus': 3, 'cumulonimbus': 4, 'altostratus': 5, 'cirrus': 6, 'stratocumulus': 7, 'cirrostratus': 8}

    features['cloudType'] = features['cloudType'].map(cloud_map) # replace using the map



    # Replace the land type strings in the dataframe with integer ids,
    # by using the map which was used when the model was compiled.
    
    # Make sure to replace the map with your model's one.
    land_map = {'Shore': 0, 'Land': 1, 'Land with Ice & Snow': 2, 'Land with lakes and rivers': 3, 'Clouds': 4, 'Desert': 5, 'Islands & Archipelagos': 6, 'Land with mountains': 7, 'none': 8}

    features['landType'] = features['landType'].map(land_map) # replace using the map



    # Remove specific colums from dataframe
    # These could be used in certain scenarios (such as when latitude doesn't cause overfitting or temperature simulates a thermal camera). 
    # Comment / uncomment if the model uses / doesn't use some of them.
    # features = features.drop('latitude', axis=1)                    # remove latitude from model input
    # features = features.drop('temperature', axis=1)                 # remove temperature from model input
    features = features.drop('precipitationIntensity', axis=1)      # remove precipitationIntensity from model input
    # features = features.drop('cloudType', axis=1)                 # remove cloudType from model input
    # features = features.drop('landType', axis=1)                 # remove landType from model input

    # Uncomment the next 2 lines to obtain the labels.txt file
    # for key in label_map.keys():
    #    print(label_map[key], key)
    

    # Run the ML model on the whole dataset
    correct_data = 0
    all_data = 0

    for i in range(0, 439):
        img = features.iloc[i].to_numpy()

        img = np.reshape(img, (1, 13))

        #print(img)


        common.input_tensor(interpreter)[:] = img
        interpreter.invoke()
        classes = classify.get_classes(interpreter, top_k=1)

        labels_from_file = read_label_file(label_file)
        
        #with open("/home/pi/rv-py-ai/airecords.csv", 'a') as file:
        #    file.write(f'{datetime.datetime.now()}, {labels.get(classes[0].id, classes[0].id)}, {classes[0].score:.5f}\n')
        
        if labels_from_file.get(classes[0].id, classes[0].id) == labels[i]:
            correct_data += 1
            #print("correct!")
        else:
            print("incorrect! expected result:", labels[i], "- actual result:", labels_from_file.get(classes[0].id, classes[0].id))
        all_data += 1
    return correct_data/all_data

print("Accuracy on the whole dataset:", analyseDataset())
