<h1>Method</h1>
  
**Image Preprocessing**

For each WSI in the adenocarcinoma dataset, respective masks for the region of interest were provided by expert pathologists. WIKM used these masks and overlayed
them onto the original WSI. We then reversed the mask such that only the normal areas of the WSI would be exposed. The motivation behind this was to create patches
of regions specific to a class (invasive, in situ, normal). 

*Masking*

Since annotated masks of invasive, in situ, and both regions were available, we decided to overlay the mask onto the WSI.
By doing this, we were able to specify the region of interest to patch in later steps. This was done simply through this function:

```
def maskOverImage(source, mask):
  source2 = cv2.resize(mask, source.shape[1::-1])

  dst = cv2.bitwise_and(source, source2)

  return dst
```

After masking the regions of interests, we reversed the mask so that we can patch from the normal areas as well.
This was done through combining the masks and ivnerting them using bitwise operations provided by OpenCV.

```
def combineAllMasks(masksList):
  for maskNum, masksPath in enumerate(masksList):
    if maskNum == 0:
      currentMask = cv2.imread(masksPath)
      prevMask = None
    else:
      currentMask = cv2.bitwise_or(cv2.imread(masksPath), currentMask)
      prevMask = currentMask

  return currentMask

combinedMasks = combineAllMasks(masksList)

combinedMasksInv = cv2.bitwise_not(combinedMasks)
slideImg = cv2.imread(slideDict[slideKey]['imgPath'])
normal = cv2.bitwise_and(slideImg, combinedMasksInv)
```


*[Patching](https://pypi.org/project/patchify/)*


Through resampling, patchify can split images into small overlappable patches by given patch cell size, and merge patches into original image.
The patchify library was only used on the adenocarcioma dataset because of its huge size.

The masked images were read through `cv2.imread()` and by using the third party function, patchify, we obtained our patches. WIKM also insantiated in the loop to ignore completely black patches.

```
size = n

image = cv2.imread(path_to_image)
patches = patchify(image, (size, size, 3), step = size)

for i in range(patches.shape[0]): #0 is first dimension
  for j in range(patches.shape[1]): #1 is second dimension
    singlePatch = patches[i, j, 0, :, :, :]

    if not np.all(singlePatch == 0): #ignore all black patches
      cv2.imwrite(f'Patch{patchNum}.png', singlePatch)
      patchNum += 1
```

WIKM tried 3 different dimensionalities and found the following:

- 100 x 100: huge loss in resolution (unusable)
- 256 x 256: satisfactory 
- 512 x 512: high resolution, computationally expensive 

We decided to mostly use the 256 x 256 patches for training and occasionally tested on the 512 x 512 patches if it was computationally viable. 

There were an overwhelmingly large batch of patches that were either all white or black due to its relatively small size 
compared to the original image. This would be too much noise for the training models so we created a function that deleted 
all patches with a mean RGB value of μ < 10 and μ > 240. 



*Numpy Array*

Due to overfitting problems, WIKM decided to revist the preprocessing stage and use numpy arrays instead of the direct image for training. The notebook for 
preprocessing can be accessed [here](https://colab.research.google.com/drive/1BqdsOIpxUPujidFBcp_xyrjTH0C4KDze?usp=sharing).

The first step was to change all of our 256x256 patches to numpy arrays:

```
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB")
    for IMAGE_NAME in tgdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == '.jpg':
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE))
            IMG.append(np.array(img))
    return IMG
```

We then split the numpy arrays by each class (i.e. invasive, insitu, or normal) and set (i.e. train or test). 
Each class label was denoted by an array of zeros, ones, etc. The training data was concatenated to X_train and the labels were concatenated to Y_train.
After saving the arrays to your local drive, load it as X_train and Y_train. Lastly, we randomly shuffled the data like so:

```
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]
```

This preprocessing method turned out to be much more efficient and resulted in much better validation accuracy and lower validation loss. 


*Dataset Split*


We then split the adenocarcioma and osteosarcoma dataset into Train, Validation, and Test sets (80-10-10). 
In the respective splits, we also separated the patches by class; Normal, In Situ, and Invasive for Adenocarcinoma and Non Viable Tumor, Viable Tumor, and Non 
Tumor for Osteosarcoma


*Augmentation*


WIKM ttilized Keras’ ImageDataGenerator built-in function to perform augmentation on ‘in situ’ and ‘both’ classes. These 2 classes were specifically selected to 
create a more evenly distributed, robust dataset. Augmentations methods used but not limited to: rotation, flipping, zoom, and shift. 

The augmentation process generated around ~6000 total, new patches and increased our dataset to ~38000 patches.


**Segmentation**


{still in progress}


**Classification**


WIKM implemented transfer learning on 4 pre-trained models. You may view the notebook [here](https://colab.research.google.com/drive/1oU3L4ldJyH1MnQwPNQCW2sZ5p9CIx8iP?usp=sharing):

- VGG16
- ResNet50, 152
- DenseNet 201

WIKM also attempted to build 1 plain convolutional neural network from scratch.
