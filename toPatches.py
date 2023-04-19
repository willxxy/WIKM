import cv2
import os
import numpy as np
import argparse
import pathlib
import json

'''
This file is specifically for the Adenocarcinoma (Lung Cancer) dataset.
It will generate invasive, in-situ, normal patches from the 25 whole 
slide images. Download the data from link in readme before running this.
'''
#========== helper functions ==========
def maskOverImage(source, mask): # put mask over image
    source2 = cv2.resize(mask, source.shape[1::-1])

    dst = cv2.bitwise_and(source, source2)

    return dst

def combineAllMasks(masksList):
    for maskNum, masksPath in enumerate(masksList):
        if maskNum == 0:
            currentMask = cv2.imread(masksPath)
            prevMask = None
        else:
            currentMask = cv2.bitwise_or(cv2.imread(masksPath), currentMask)
            prevMask = currentMask

    return currentMask


#========== create folder structure ==========
def structure(dataPath):
    pathDataset = []
    patientDirs = [str(pathlib.Path(dataPath, patient)) for patient in os.listdir(dataPath) if patient[:7]=="LungFCP"]
    
    for patientNumber, patientDir in enumerate(patientDirs, 1):

        slidesImages = sorted(os.listdir(str(patientDir) + '/images'))
        maskImages = sorted(os.listdir(str(patientDir) + '/annotations'))

        slideDict = {}

        for slideNum, images in enumerate(slidesImages, 1):
            slideKey = f'slide{slideNum}'
            slideDict[slideKey] = {}
            slideDict[slideKey]['imgPath'] = str(pathlib.Path(str(patientDir), 'images', images).absolute())
            slideDict[slideKey]['invasive'] = list()
            slideDict[slideKey]['in_situ'] = list()
            slideDict[slideKey]['both'] = list()
            slideDict[slideKey]['airway'] = ""
            slideDict[slideKey]['blood'] = ""
            annotsForCurrSlide = [annot for annot in maskImages if annot[:18] == images[:-5]]

            for annotations in annotsForCurrSlide:
                pathToAnnotation = str(pathlib.Path(patientDir, 'annotations', annotations).absolute())
                serial = annotations[33:-5]

                if serial in ["R000G000B255","R001G000B255","R002G000B255"]: #in situ
                    slideDict[slideKey]['in_situ'].append(pathToAnnotation)
                elif serial in ["R000G255B000", "R001G255B000", "R002G255B000", "R003G255B000", "R004G255B000", "R005G255B000", "R006G255B000"]: #invasive
                    slideDict[slideKey]['invasive'].append(pathToAnnotation)
                elif serial == "R255G000B000": #both region
                    slideDict[slideKey]['both'].append(pathToAnnotation)

            patientDirPathObj = pathlib.Path(patientDir)
            airPath = pathlib.Path(patientDirPathObj.parent, 'FinalPublishedResults', patientDirPathObj.name, 'histology', 'masks', 'airways')  #some don't have
            bloodPath = pathlib.Path(patientDirPathObj.parent, 'FinalPublishedResults', patientDirPathObj.name, 'histology', 'masks', 'blood')

            if os.path.exists(airPath):
                for air in os.listdir(airPath):
                    if images[:-1].lower() == air.lower() or images[:-5].lower() == air[:-4].lower(): #some tif some png
                        slideDict[slideKey]['airway'] = str(pathlib.Path(airPath, air).absolute()) # os.path.join(airPath, air)

            if os.path.exists(bloodPath):
                for bld in os.listdir(bloodPath):
                    if images[:-1].lower() == bld.lower() or images[:-5].lower() == bld[:-4].lower():
                        slideDict[slideKey]['blood'] = str(pathlib.Path(bloodPath, bld).absolute()) # os.path.join(bloodPath, bld)

        patientDict = {}
        patientDict['patient{}'.format(patientNumber)] = slideDict
        pathDataset.append(patientDict)
    return pathDataset


def patching(pathDataset, patchFolder, slideWindw, patchSize, threshold, maskOut):
    # create label folders
    if not os.path.exists(f'{patchFolder}/in_situ'):
        os.makedirs(f'{patchFolder}/in_situ')
    if not os.path.exists(f'{patchFolder}/invasive'):
        os.makedirs(f'{patchFolder}/invasive')
    if not os.path.exists(f'{patchFolder}/normal'):
        os.makedirs(f'{patchFolder}/normal')
    # getting paths to each class
    invasive = []
    in_situ = []
    normal = []
    for patient in pathDataset:
        for pat,infoDict in patient.items():
            for slide,paths in infoDict.items():
                both = []
                if paths['invasive'] != []:
                    invasive.append((paths['imgPath'],paths['invasive']))
                    both += paths['invasive']
                if paths['in_situ'] != []:
                    in_situ.append((paths['imgPath'],paths['in_situ']))
                    both += paths['in_situ']
                if both != []:
                    normal.append((paths['imgPath'],both))
    tempDict = {}
    tempDict['invasive'] = invasive
    tempDict['in_situ'] = in_situ
    slideW = patchSize if slideWindw == 0 else slideWindw
    
    #========== Patching from Invasive, In Situ regions ==========
    for label in tempDict:
        count = 0
        print(f'=================={label}================')
        for img,lstOfMk in tempDict[label]:
            print("processing",img)
            slide = cv2.imread(img)
            if lstOfMk == []:
                continue
            for mk in lstOfMk:
                mask = cv2.imread(mk)

                y_min, x_min = 0, 0
                y_max, x_max = patchSize, patchSize

                # --- calculate number of iterations for y and x axis
                y_itrs = int(slide.shape[0] / patchSize)
                x_itrs = int(slide.shape[1] / patchSize)

                for y in range(y_itrs):
                    for x in range(x_itrs):
                        slide_patch = slide[y_min:y_max, x_min:x_max, :]
                        mask_patch = mask[y_min:y_max, x_min:x_max, :]

                        # --- check if majority of cropped label is label of interest        
                        flat = mask_patch.flatten()
                        roi = np.where(flat==255)
                        if flat.size == 0:
                            continue
                        percentage = roi[0].size / flat.size
                        if percentage > threshold:
                            #black out non-ROI region
                            newSlide = cv2.bitwise_and(slide_patch,mask_patch)

                            #filter almost all black, all white patches
                            temp = np.copy(newSlide)
                            rgb_mean = np.mean(temp)
                            if (rgb_mean > 230) or (rgb_mean < 10):
                                continue


                            path = f'{patchFolder}/{label}/patch{count}.png'
                            if maskOut:
                                cv2.imwrite(path, newSlide)
                            else:
                                cv2.imwrite(slide_patch)
                            count += 1

                        x_min += slideW
                        x_max += slideW

                    # --- reset x_min and x_max
                    x_min = 0
                    x_max = patchSize

                    y_min += patchSize
                    y_max += patchSize
    
    #========== Patching from normal regions (reverse of invasive, in_situ) ==========
    count = 0
    print(f'==================normal================')
    for img,lstOfMk in normal:
        print("processing",img)
        if lstOfMk == []:
            continue
        slide = cv2.imread(img)
        combineMasks = combineAllMasks(lstOfMk)
        combinedMasksInv = cv2.bitwise_not(combineMasks)
        if combinedMasksInv.shape != slide.shape:
            print("not the same size!")
            continue

        y_min, x_min = 0, 0
        y_max, x_max = patchSize, patchSize

        # --- calculate number of iterations for y and x axis
        y_itrs = int(slide.shape[0] / patchSize)
        x_itrs = int(slide.shape[1] / patchSize)

        for y in range(y_itrs):
            for x in range(x_itrs):

                slide_patch = slide[y_min:y_max, x_min:x_max, :]
                mask_patch = combinedMasksInv[y_min:y_max, x_min:x_max, :]

                # --- check if majority of cropped label is label of interest        
                flat = mask_patch.flatten()
                roi = np.where(flat==255)
                if flat.size == 0:
                    continue
                percentage = roi[0].size / flat.size
                if percentage > threshold:
                    #black out non-ROI region
                    newSlide = cv2.bitwise_and(slide_patch,mask_patch)

                    #filter almost all black, all white, all grey, weird marks patches
                    temp = np.copy(newSlide)
                    rgb_mean = np.mean(temp)
                    if (rgb_mean > 230) or (rgb_mean < 10) or (rgb_mean == None):
                        continue
                        
                    path = f'{patchFolder}/normal/patch{count}.png'
                    if maskOut:
                        cv2.imwrite(path, newSlide)
                    else:
                        cv2.imwrite(slide_patch)
                    count += 1

                x_min += slideW
                x_max += slideW

            # --- reset x_min and x_max
            x_min = 0
            x_max = patchSize

            y_min += patchSize
            y_max += patchSize


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path', type=str, help="folder path to lung cancer dataset (root dir)")
    parser.add_argument('--patchesPath', default="patchedPNG", help="folder to save patches")
    parser.add_argument('--slideWindw', default=0, help="sliding window size for patching")
    parser.add_argument('--patchSize', default=256)
    parser.add_argument('--roi', default=0.1, help="percent of ROI for a patch to be saved")
    parser.add_argument('--maskOut', default=True, help="whether to mask out only the specified label when patching")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.path):
        structPaths = structure(args.path)
        # print(json.dumps(structPaths, indent = 5))
        patching(structPaths, args.patchesPath, args.slideWindw, args.patchSize, args.roi, args.maskOut)

    else:
        print("Below path doesn't exist:")
        print(os.path.abspath(args.path))