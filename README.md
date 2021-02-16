# DeepMergeDomainAdaptation
Deep domain adaptation for astronomy using Maximum Mean Discrepancy (MMD) and adversarial training using Domain Adversarial Neural Networks (DANNs)  - simulated and observer galaxy mergers. We also implement optional use of the Fisher loss and Entropy minimization to enforce within class compactness and between class separability in both domains.
![](images/source_target.png)

### About
Working with images from two domains is a challanging problem in all domains of computer vision. In astronomy, transferability of deep learning models from simulated to real observed images or from one telescope to another, is a necessity. 
Drop in model classification accuracy between images used to train the model (source domain) and images from some other domain (target domain) can be mitgated to some extent by using domain adaptation techniques. 
Here we implement domain adaptation to improve classification of simulated and real merging galaxies. Studies of merging galaxies are challending due to small number of identified mergers in real data, biases in labeling those objects, as well as differences between observed and simulated objects. Domain adaptation and similar techniques will in the future be needed in order to sucessfully transfering knowledge obtained from these different domains. They will also play a crutial role for sucessfull detection of new merger candidates in very large astronomical surveys that will be available in the near future.

### Architecture
Our experiments were performed using a simple CNN architecture (3 convolutional, 3 pooling adn 3 danse layers) called DeepMerge, as well as using more complex and well known ResNet18 network.

### Prepare Datasets
Images used can be found at XXXXXXXXXXXXXX. There are two tipes of experiments we run: Simulation to Simulation; and Simulation to Real.

#### Simulation to Simulation Experiments
Experiments study distant merging galaxies using simulated images from Illustris-1 cosmological simulation. Images have 3 filters that mimic Hubble Space Telescope (HST) observations (ACS F814W,NC F356W, WFC3 F160W).

Source Domain - Illustris-1 images (snapshot z=2) with added PSF mimicking HST observations

     - Images: SimSim_SOURCE_X_Illustris2_pristine.npy
     
     - Labels: SimSim_SOURCE_y_Illustris2_pristine.npy

Target Domain -  Illustris-1 images (snapshot z=2) with added PSF and observational noise mimicking HST

    - Images: SimSim_TARGET_X_Illustris2_noisy.npy
    
    - Labels: SimSim_TARGET_y_Illustris2_noisy.npy

#### Simulation to Real Experiments
Experiments study nearby merging galaxies using simulated Illustris-1 images and real Sloan Digital Sky Survey (SDSS) images. All images have 3 filters. SDSS images have (g,r,i) filters, while simulated Illustris images also mimic the same 3 filters.

Source Domain - Illustris-1 images (snapshot z=0) with added effects of dust, PSF and observational noise mimicking SDSS observations

    - Images: SimReal_SOURCE_X_Illustris0.npy
    
    - Labels: SimReal_SOURCE_y_Illustris0.npy

Target Domain - SDSS images of merging galaxies, labeled through Galaxy Zoo project

    - Images: SimReal_TARGET_X_postmergers_SDSS.npy
    
    - Labels: SimReal_TARGET_y_postmergers_SDSS.npy


### Training
Explanation how to run different versions of training as well as evaluation are given in example notebook: Running_DeepMergeDomainAdaptation_files.ipynb 


### Requirements
Code was developed using Pytorch 1.7.0. The requirements.txt file should list all Python libraries that your notebooks depend on, and they can be installed using:
```
pip install -r requirements.txt
```


### Authors
- Aleksandra Ćiprijanović
- Diana Kafkes
- Gabriel Perdue

### References
If you use this code, please cite our paper: arXiv:..........
