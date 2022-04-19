# Crowd Social-Distancing Classification

## CS 665 Final Project

Seth Lewis, Nick Netterville, Carmen Saunders, Laura Thompson  
CS 665, Deep Learning  
SPR 2022, 04/21/2022

## Objective

The goal of this project is to train a binary neural network classifier on images of crowds to identify whether members of the crowd are socially distancing or not. CNN transfer learning was used to accomplish this goal.

## Setup

Under the assumption that you have Anaconda 3 installed on your system [(installing anaconda)](https://www.anaconda.com/products/distribution?gclid=CjwKCAjwu_mSBhAYEiwA5BBmfygpN8njjMuRGfmwN17uqSMgM7zG2qGecD4u63Sn8kD5-VmKbLyFdBoCP3wQAvD_BwE). From the main folder of this repository run: 

```
conda activate
conda env create
```

## Methodology

* Data collection: data for the non distanced classes came from the jhu v.2.0 crowd dataset found here [crowd counting data](http://www.crowd-counting.com/), the didstanced classes we collected using a google image search for social distancing and then continued refinement to include additional search terms for locations and activities such as; dance, restaurant, concert, airport, etc. Image selection was very conservative in that only images that had clear indication of social distancing measures were used, if there was any doubt about the image it was removed. Furthermore image duplicates were removed using the provided notebook [duplicate_image_remove.ipynb](duplicate_image_remove.ipynb)

* Model selection: for the purposes of training a robust model in the least amout of time, transfer learning was performed using ResNet50, VGG16, and a combination of the features of each the associated Keras applications were used in the implementation [https://keras.io/api/applications/](https://keras.io/api/applications/)

* Data splitting: a 2:1 negative to positive class imbalance was maintained due to the scarceness of positive cases through all splits. Training to validation was numerically representative to a 80/20 split of the total images found in both sets.

* Visualizing the results: To accomodate visualization and results explanation 3 convolutional layers and one multiply layer were added to the model to form a proxy attention network. Images were then overlaid with the output of these layers to show what parts of any given image the combined filters at that layer were looking at. Additionally, a set of activation maps were created to for these new layers to further visualize the activation functions on a per filter basis. All of this visualization was done on test data.

* Evaluation: metric based evaluation uses confusion matrices, recall, precision, and F1 measure. Recall, precision, and F1 were calculated for both classes of the binary classifier and can be found in the accompanying notebooks, [single_model_transfer.ipynb](single_model_transfer.ipynb) and [combined_features_transfer.ipynb](combined_features_transfer.ipynb) 

## To Use this Repository

Download the associated zip from github and follow the setup instructions above. If you wish to run the notebooks it is best to start with the single transfer and combined features transfer notebooks and see how the code is executed. If you wish to experiment with your own data then it is suggested to collect a set of images numbering 500 for the distanced class and place them in the positive_cases folder. Follow this by running the duplicate removal notebook and the data generation notebook excluding the negative case section unless you would like to supply the jhu dataset at [crowd counting data](http://www.crowd-counting.com/)

## Results, metrics and model summaries

All output including images of the attention layers output, activation maps, texts of the model summaries, and csv files of the results can be found [here](output/images/) and [here](output/dataframes/)
