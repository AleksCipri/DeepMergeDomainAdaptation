# DeepMergeDomainAdaptation
Fisher deep domain adaptation for astronomy - galaxy mergers. 
![](images/source_target.png)

### Abstract
Working with images from two domains is a challanging problem in all domains of computer vision. In astronomy, transferability of deep learning models from simulated to real observed images or from one telescope to another, is a necessity. 
Drop in model classification accuracy between images used to train the model (source domain) and images from other sources (target domain) can be mitgated to some extent by using domain addaptation techniques or domain adversarial training. 
Here we try to implement these techniques to show and improve image classification of simulated distant merging galaxies, with the goal that these techniques will in the future be used for transfering knowledge obtained from simulated galaxies to real observations from future large scale survays, 
which will produce bigger distant merging galaxy datasets.

### Architecture
DeepMerge CNN has 3 convolutional layers (with 3 pooling layers) and 3 dense layers, added after flattening. It also has dropout after each convolutional layer, as well as weight regularization in the first two dense layers.
We also implement ResNetXXX...

### Prepare Datasets
Images used can be found at https://doi.org/10.17909/t9-vqk6-pc80. Pristine and noisy images used in the paper can be found in SB00_augmented_3FILT.npy and SB25_augmented_3FILT.npy, respectively (label files: SB00_augmented_y_3FILT.npy and SB25_augmented_y_3FILT.npy). Raw (large) and resized images are also available for those who would like to try their own image formating and augmentation. Images we use have 3 filters, which mimic those available onboard the Hubble Space Telescope.

### Training
To run training use .....


### Requirements
Code was developed using Python XXX. The requirements.txt file should list all Python libraries that your notebooks depend on, and they can be installed using:
```
pip install -r requirements.txt
```


### Authors
- Aleksandra Ćiprijanović
- Diana Kafkes
- Gabriel Perdue

### References
If you use this code, please cite our paper: .....
