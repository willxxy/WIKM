import cv2
import os
import numpy as np
import random
import argparse
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--modelPath', type=str, default='models', help='folder path to model')
    parser.add_argument('--dataset', type=str, default="bone", help='which cancer dataset to test on (bone/lung)')
    parser.add_argument('--dataPath', type=str, default="data", help='folder path to data')
    parser.add_argument('--count', type=int, default=1, help='how many patches to visualize and predict')
    parser.add_argument('--size', type=int, default = 256, help = 'size of patches')
    return parser.parse_args()

def predictBone(args):
    boneModel = load_model(f'{args.modelPath}')
    # classes: non-tumor: 0, non-viable tumor: 1, viable: 2
    if not os.path.exists(f'{args.dataPath}/OsteosarcomaData'):
        print("Below path doesn't exist: (download Osteosarcoma Bone Data to the data folder)")
        print(os.path.abspath(f'{args.dataPath}/OsteosarcomaData'))
        return
    classes = os.listdir(f'{args.dataPath}/OsteosarcomaData')
    if classes == 0:
        print("No class folders found.")
        return
    labels = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable']
    for i in range(args.count):
        randomClass = random.choice(classes)
        patches = os.listdir(f'{args.dataPath}/OsteosarcomaData/{randomClass}')
        randomPatch = random.choice(patches)
        img = np.asarray(Image.open(f'{args.dataPath}/OsteosarcomaData/{randomClass}/{randomPatch}').convert("RGB"))
        img = cv2.resize(img, (args.size, args.size))
        plt.imshow(img)
        plt.title(f'True label: {randomClass}')
        print(f'True label: {randomClass}')

        img = tf.expand_dims(img, axis=0) #=> (1, 224, 224, 3)
        yhat = boneModel.predict(img)
        prediction = labels[np.argmax(yhat)]
        plt.figtext(0.5, 0.005, f'Model chooses: {prediction}\n', ha="center", color='green')
        plt.show()
        print(prediction)
        

def predictLung(args):
    lungModel = load_model(f'{args.modelPath}')
    # classes: non-tumor: 0, non-viable tumor: 1, viable: 2
    if not os.path.exists(f'{args.dataPath}'):
        print("Below path doesn't exist: (download Adenocarcinoma Lung Data to the data folder)")
        print(os.path.abspath(f'{args.dataPath}'))
        return
    classes = os.listdir(f'{args.dataPath}')
    if classes == 0:
        print("No class folders found.")
        return
    labels = ['normal', 'invasive', 'in-situ']
    randomClass = random.choice(classes)
    if '.npy' in os.listdir(f'{args.dataPath}/{randomClass}'):
        for i in range(args.count):
            randomClass = random.choice(classes)
            insitu_np = np.load(f'{args.dataPath}/{randomClass}/insituPatch.npy')
            invasive_np = np.load(f'{args.dataPath}/{randomClass}/invasivePatch.npy')
            normal_np = np.load(f'{args.dataPath}/{randomClass}/normalPatch.npy')
            patches = np.concatenate((insitu_np, invasive_np, normal_np))
            randomPatch = random.choice(patches)
            img = cv2.resize(randomPatch, (args.size, args.size))
            plt.imshow(img)
            plt.title(f'True label: {randomClass}')
            print(f'True label: {randomClass}')

            img = tf.expand_dims(img, axis=0) #=> (1, 224, 224, 3)
            yhat = lungModel.predict(img)
            prediction = labels[np.argmax(yhat)]
            plt.figtext(0.5, 0.01, f'Model chooses: {prediction}\n', ha="center")
            plt.show()
            #print(yhat)
            print(prediction)
    else:        
        for i in range(args.count):
            randomClass = random.choice(classes)
            patches = os.listdir(f'{args.dataPath}/{randomClass}')
            randomPatch = random.choice(patches)
            img = np.asarray(Image.open(f'{args.dataPath}/{randomClass}/{randomPatch}').convert("RGB"))
            img = cv2.resize(img, (args.size, args.size))
            plt.imshow(img)
            plt.title(f'True label: {randomClass}')
            print(f'True label: {randomClass}')

            img = tf.expand_dims(img, axis=0) #=> (1, 224, 224, 3)
            yhat = lungModel.predict(img)
            prediction = labels[np.argmax(yhat)]
            plt.figtext(0.5, 0.01, f'Model chooses: {prediction}\n', ha="center")
            plt.show()
            #print(yhat)
            print(prediction)

if __name__ == "__main__":
    args = get_args()
    if args.dataset == 'bone':
        print("==== Evaluating model on Osteosarcoma (Bone Cancer) ====")
        predictBone(args)
    elif args.dataset == 'lung':
        print("==== Evaluating model on Adenocarcinoma (Lung Cancer) ====")
        predictLung(args)
