![Image](./resources/vippdiism.png)

# Multi-Branch CNN Copy Move Source-Target Disambiguation

This repository is an implementation of the method in 

"Copy Move Source-Target Disambiguation through Multi-Branch CNNs" 
Mauro Barni, Quoc-Tin Phan, Benedetta Tondi

The disambiguation method allows to identify source and target region of a copy-move by means of a multi-branch 
CNN-based architecture, capable to learn suitable features by looking at the presence of interpolation  artifacts 
and boundary inconsistencies.

Code implemented by Quoc-Tin Phan (dimmoon2511[at]gmail.com)


## Step 1. Prerequisites

All prerequisites should be listed in requirements.txt.

To avoid conflicts with your system, it is encouraged to first create virtual environment in Anaconda.

Install all prerequisites:

```pip install -r resources/requirements.txt```

## Step 2. Download pre-trained models

Download [pre-trained models](https://drive.google.com/file/d/1LS2Z8bLnL_non7Ibcl9QWd1zBGl_PAD6)
and unzip the folder.

## Step 3 (optional). Training

To generate synthetic datasets use the Matlab scripts in ./resources/db_gen:

* For training 4-Twins net: ```run_create_scribble_db.m```.

* For training Siamese net: ```run_create_scribble_rigid_cm_db.m```

** Added on 17/05/2021 **. 

The list of images used to train the networks is now available [here] (https://github.com/andreacos/MultiBranch_CNNCopyMove_Disambiguation/blob/master/resources/db_gen/TRAINING_SET.txt). To use it, proceed as follows:  
* download the three required datasets: RAISE2K, VISION and Dresden from their respective websites; 
* use the list to move only the training images into a single folder; 
* edit the [Matlab db generation code](https://github.com/andreacos/MultiBranch_CNNCopyMove_Disambiguation/blob/master/resources/db_gen/run_create_scribble_rigid_cm_db.m) to point to that folder (e.g. this line of code for the rigid case: `input_path  = '../datasets/RAISE_DRESDEN_VISION_mixed')`


To train 4-Twins net, re-configure parameters in **train\_4twins.py**:

* ```pos_neg.txt```: path to the file whose each line contains absolute path to a training image.
* ```max_items```: the number of training images (the number of lines in ```pos_neg.txt``` should be equal or over this number).
* ```use_gpu```: the list of GPU's ID that we use. A batch of data is equally split and distributed on each GPU. The final loss is synthesized from each GPU.
* ```working_dir```: where to save the model and training logs.
* ```pretrained_resnet```: directory of resnet50 pretrained on ImageNet (you can download from [here](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz), and decompress into this directory).


Similarly, re-configure parameters in **train\_siamese.py** to train Siamese net:

* ```pos_lst.txt```: path to the file whose each line contains absolute path to a training image. 
* Other parameters can be configured analogously to 4-Twins net.

## Step 4. Prediction

Execute:

```python predict_4twins.py```

```python predict_siamese.py```

```python predict_fusion.py```
