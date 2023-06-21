# ClimateClassificationML
This repository includes the code used to train and test our climate classification machine learning model, using Keras. This requires a dataset generated with [code from this repository](https://github.com/Team-Octans-AstroPi/climateCSVgenerator).

### Data from <picture><source media="(prefers-color-scheme: dark)" srcset="https://weatherkit.apple.com/assets/branding/en/Apple_Weather_wht_en_3X_090122.png"><source media="(prefers-color-scheme: light)" srcset="https://weatherkit.apple.com/assets/branding/en/Apple_Weather_blk_en_3X_090122.png"><img src="" height="18" alt="Apple Weather Logo"></picture>
The dataset used by this model, generated using [this code](https://github.com/Team-Octans-AstroPi/climateCSVgenerator), uses data from Apple Weather, that may be modified and used for training the ML model in other repositories.
Data sources attribution: https://developer.apple.com/weatherkit/data-source-attribution/.

### Data from climateapi.scottpinkelman.com
The dataset used by this model, generated using [this code](https://github.com/Team-Octans-AstroPi/climateCSVgenerator), uses data from http://climateapi.scottpinkelman.com.
It uses data from the Institute for Veterinary Public Health and the Provincial Government of Carinthia in Austria.

<b>Citation:</b><br>
Kottek, M., J. Grieser, C. Beck, B. Rudolf, and F. Rubel, 2006: World Map of the Köppen-Geiger climate classification updated. Meteorol. Z., 15, 259-263. DOI: 10.1127/0941-2948/2006/0130.

## Repository Contents
- `tfclimatemodel.py` is the Tensorflow code used to make and train our model. It also saves this model as a `.tflite` to be able to test it on a Raspberry Pi with Coral Edge TPU.
- `coral-test-rpi.py` is a program designed to run on a Raspberry Pi with a Coral Edge TPU, which tests the whole data

## Model structure
Here is the model's summary, printed by Tensorflow:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 64)                896       
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 64)                4160      
                                                                 
 dense_3 (Dense)             (None, 16)                1040      
                                                                 
=================================================================
Total params: 10,256
Trainable params: 10,256
Non-trainable params: 0
_________________________________________________________________
```

## Model Performance
We achieved 97% validation accuracy, and 77% accuracy on new test data.

As shown in the graph below, on our dataset, around 450 epochs, the learning rate stalled. This is why we limited the model to 450 epochs.<br><br>
![validation accuracy and loss graph](https://github.com/Team-Octans-AstroPi/climateClassificationML/assets/80255379/cd8b2031-e120-4083-a389-d08fc161ea89)
