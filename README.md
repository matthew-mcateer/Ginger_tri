# `Ginger_Tri` Dataset: Image Dataset for 3 tuberous rhizomes, and initial experiments with Transfer Learning
This project is a result of a collaboration with the team at [UnifyID](https://unify.id/). Pull Requests and Issue flags are more than welcome.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Generic badge](https://img.shields.io/badge/Status-Active-<COLOR>.svg)](https://shields.io/)

## Getting Started

1. Clone this repo using `https://github.com/matthew-mcateer/Ginger_tri.git` (or follow the Google Colab links below for the GPU-enabled, in-browser instances).
2. Unzip the `.zip` files with the image resolution of interest
    
3. To work with prexisting models, download the `.log` and `.hdf5` files from the links below: 

| Architecture 	| Train Acc. 	| Val Acc. 	| Model(`.hdf5` file)   	| Training Logs (`.log` file) 	| Best Version (`.hdf5` file) 	|
|--------------	|------------	|----------	|-----------------------	|-----------------------------	|-----------------------------	|
| InceptionV3  	| 97.51%      	| 99.87%    	| [Google Drive Link](https://drive.google.com/file/d/11PwyV7bgFE16rE1HsUCHbMqfdBRgg3PI/view?usp=sharing) 	| [Google Drive Link](https://drive.google.com/file/d/1EBekmHQRahzoFv3xE-pInhts0XvwIG1L/view?usp=sharing)       	| [Google Drive Link](https://drive.google.com/file/d/101YsYoOqhF7DHqfQUHJ9lqMFjZr8vVuA/view?usp=sharing)       	|
| NASNetMobile       	| 94.98%      	| 89.20%    	| [Google Drive Link](https://drive.google.com/file/d/1NUVRCgkhsgbDl9Sdm31gxS5q5KkERB3H/view?usp=sharing) 	| [Google Drive Link](https://drive.google.com/file/d/1G3gPKxALb4WVSyb2e7a8j_l0gDMavugH/view?usp=sharing)       	| [Google Drive Link](https://drive.google.com/file/d/1tCkiPduW5GQmwRd_j52tv7xZnul48rM4/view?usp=sharing)       	|

## Project Description
This repo provides a new dataset for object recogntion, as well as a framework for running and testing transfer learning strategies.

Transfer learning is a useful technique in machine learning. It is especially prevalent in image- and facial-recognition tasks, where training a classifier on a relatively small amount of data can be bolstered by models that have been pre-trained on larger tasks.[[1]](#scrollTo=ek-3esN_z32w)

In this demonstration, we explore the use of transfer learning (combined with random cropping for data augmentation) as a method for a few-shot classification task where the classes are extremely similar (or in our case, barely distinguishable to many humans). The random-cropping is important for ou particular analysis, as it makes the classification task more reliant on texture-classification. When using convolutional neural networks, or any network that relies on convolution filters, one is implicitly assuming Spatial Homogeneity (SH). This means the first-order statistics of the filter/neuron activations are independent of their spatial location in the image sample. 


## Featured Notebooks/Analysis/Deliverables

There are several steps to the experimental workflow for this project:

For the results of these steps, see the notebooks below

| Subject                	| Jupyter notebook Link 	| Google Colab Link 	|
|---------------------------------	|:----------------------:|------------------	|
| Dataset Creation   (Steps 1, 2, & 3)             	| [JuPyter Notebook](https://github.com/matthew-mcateer/Ginger_tri/blob/master/Dataset_Preparation.ipynb) 	| [Google Colab](https://colab.research.google.com/github/matthew-mcateer/Ginger_tri/blob/master/Dataset_Preparation.ipynb) 	|
| FineTuning with InceptionV3 (Step 4)    	| [JuPyter Notebook](https://github.com/matthew-mcateer/Ginger_tri/blob/master/TransferLearning_InceptionV3_FineTuning.ipynb) 	| [Google Colab](https://colab.research.google.com/github/matthew-mcateer/Ginger_tri/blob/master/TransferLearning_InceptionV3_FineTuning.ipynb) 	|
| FineTuning with NASNetMobile  (Step 4)    	| [JuPyter Notebook](https://github.com/matthew-mcateer/Ginger_tri/blob/master/TransferLearning_NasNetMobile_FineTuning.ipynb) 	| [Google Colab](https://colab.research.google.com/github/matthew-mcateer/Ginger_tri/blob/master/TransferLearning_NasNetMobile_FineTuning.ipynb) 	|
| Transfer Learning Visualization  (Step 5)  	| [JuPyter Notebook](https://github.com/matthew-mcateer/Ginger_tri/blob/master/Transfer_learning_Visualization.ipynb) 	| [Google Colab](https://colab.research.google.com/github/matthew-mcateer/Ginger_tri/blob/master/Transfer_learning_Visualization.ipynb) 	|

## Comparison to Other Datasets

For the sake of our experiments, we will be focusing on using networks that were pre-trained with weights from ImageNet as a classifier for our new Ginger-Tri dataset. Below is a side-by-dise comparison of the two datasets:

|                      	|                ImageNet               	|                Ginger-Tri               	|
|-------------------	|:-------------------------------------:	|:---------------------------------------:	|
| classes           	| 20,000                                	| 3                                       	|
| sources           	| all over internet                     	| a Whole Foods produce section           	|
| instances         	| 14,197,122                            	| 3000                                    	|
| size              	| 150 GB                                	| 24.2 MB                                 	|
| Created (updated) 	| 2009 ( 2014 )                           	| 2019                                    	|
| Format            	| Images                                	| Images                                  	|
| Default Task      	| Object recognition, scene recognition 	| Object recognition, texture recognition 	|
| Authors           	| J. Deng et al.                        	| V. Prabhu et al.                          	|

The images in the Ginger_tri dataset are available in [64 x 64 Pixel](https://github.com/matthew-mcateer/Ginger_tri/raw/master/ginger_tri_64x64.zip), [128 x 128 Pixel](https://github.com/matthew-mcateer/Ginger_tri/raw/master/ginger_tri_128x128.zip), [256 x 256 Pixel](https://github.com/matthew-mcateer/Ginger_tri/raw/master/ginger_tri_256x256.zip), and [512 x 512 Pixel](https://github.com/matthew-mcateer/Ginger_tri/raw/master/ginger_tri_512x512.zip) formats.

## Future Work

- Additional Data Augmentation Steps
- Hyperparameter Optimization (especially for NASNetMobile)
- Fine-tuning models with additional classes, more images of the classes, or both
- Comparison to Random initialization, as several pieces of literature claim that this is an acceptable substitute for pre-trained models.
- Comparison to InceptionV3 trained from scratch on this dataset (accuracy and loss over time for the same number of epochs with both models)
- Applying other tools for interpretability (such as Flame or Darkon or Lucid)

## References

### Papers

[1] Do Better ImageNet Models Transfer Better? https://arxiv.org/pdf/1805.08974.pdf

[2] A Kronecker-factored approximate Fisher matrix for convolution layers http://proceedings.mlr.press/v48/grosse16.pdf

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). [Dropout: a simple way to prevent neural networks from overfitting.](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Zeiler, M. D., & Fergus, R. (2014, September). [Visualizing and understanding convolutional networks.](https://arxiv.org/abs/1311.2901) In European conference on computer vision (pp. 818-833). Springer, Cham. 

[5] Rethinking ImageNet Pre-training. https://arxiv.org/abs/1811.08883

[6] Differentiable Parameterizations. https://distill.pub/2018/differentiable-parameterizations/

[7] The Building Blocks of Interpretability. https://distill.pub/2018/building-blocks/

[8] Feature Visualization. https://distill.pub/2017/feature-visualization/

[9] Understanding Black-box Predictions via Influence Functions. https://arxiv.org/pdf/1703.04730v2.pdf


### Code References

- Building powerful image classification models using very little data. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

- How convolutional neural networks see the world. https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

-  ImageTransfer Learning Tutorial from Tensorflow https://www.tensorflow.org/r2/tutorials/images/transfer_learning

- Keras Image Preprocessing https://keras.io/preprocessing/image/

## Contributors

**Team Members and Contributors:** 

|Name/Github Link     |  Twitter Handle   | 
|---------|-----------------|
|[Matthew McAteer](https://github.com/matthew-mcateer)| [@MatthewMcAteer0](https://twitter.com/MatthewMcAteer0)       |
|[Vinay Prabhu](https://github.com/vinayprabhu) |     [@VinayPrabhu](https://twitter.com/vinayprabhu)    |


