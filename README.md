# WIKM Cancer Detection

## Abstract

WIKM will aim to create a state-of-the-art artificial intelligence program that will classify invasive, in situ, and normal cells in tissue samples stained with 
hematoxylin and eosin (H&E). While extensive research and most approaches have focused on breast cancer, WIKM wants to create an AI program that can identify lung 
cancer and ultimately generalize to other pathologies.

## Dataset

WIKM's dataset consists of 25 Whole Slide Images (WSI) extracted from a study that attempts to "map the extent of invasive adenocarcinoma onto *in vivo* lung CT." 
The dimensionality of these whole slide images varied around ~9000 x ~9000. The complete dataset consists of 11,210 CT scans and 25 WSI. These images can be pubicly 
accessed through the [cancer imaging archive](https://wiki.cancerimagingarchive.net/display/Public/Lung+Fused-CT-Pathology#398787026b5d1a16191d45879f13f76e29aa18e1)

WIKM also uses another dataset regarding Osteosarcoma as a sanity check. This dataset contains 1144 whole slide images that are 1024 x 1024 in dimensionality. These 
images can also be publicly accessed through the [cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756935).  

## Usage

Run `toPatches.py` to generate patches from whole slide images for lung cancer dataset (necessary for train/test)

Run `cleanData.py` to clean the patches data. 

Run `testModel.py` to evaluate and visualize our model prediction on both datasets. 


## Method

WIKM divides the approach into three tasks: image preprocessing, segmentation, and classification.   
More details in [details.md](details.md)


## Results 


*Using Patches*



| Dataset        | Models        | Validation Accuracy  | Validation Loss | 
| -------------- | ------------- | -------------------- | --------------- |
| Adenocarcinoma | VGG16         |       ~ 0.775        |     ~ 0.55      |
|                |               |                      |                 |
| Osteosarcoma   | ResNet50      |       ~ 0.780        |     ~ 0.80      |



*Using Numpy Arrays*



| Dataset        | Models        | Validation Accuracy  | Validation Loss | 
| -------------- | ------------- | -------------------- | --------------- |
| Adenocarcinoma | VGG16         |       ~ 0.940        |     ~ 0.20      |
|                | ResNet152     |       ~ 0.945        |     ~ 0.50      |
|                | ResNet50      |       ~ 0.949        |     ~ 0.15      |
|                | DenseNet201   |       ~ 0.960        |     ~ 0.13      |
|                |               |                      |                 |
| Osteosarcoma   | VGG16         |       ~ 0.950        |     ~ 0.24      |
|                | ResNet152     |       ~ 0.956        |     ~ 0.20      |
|                | ResNet50      |       ~ 0.967        |     ~ 0.09      |
|                | DenseNet201   |       ~ 0.961        |     ~ 0.16      |
|                | Plain CNN     |       ~ 0.790        |     ~ 0.80      |










