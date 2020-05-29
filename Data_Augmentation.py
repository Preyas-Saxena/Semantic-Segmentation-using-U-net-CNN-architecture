
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def horizontal_rotation(images_tensor):
    rotated_imgs=[None]*len(images_tensor)
    for i in range(len(images_tensor)):
        img= images_tensor[i]
        rotated_imgs[i]=horizontal_flip(img)

    rotated_features=np.array(rotated_imgs)
    return rotated_features

#Augmenting data:
def augment(imageset, labelset):

    himages = horizontal_rotation(imageset)
    hmasks = horizontal_rotation(labelset)

    vimages = horizontal_flip(imageset)
    vmasks = horizontal_flip(labelset)

    aug_images = np.concatenate((imageset, himages, vimages), axis=0)
    aug_masks = np.concatenate((labelset, hmasks, vmasks), axis=0)

    return aug_images, aug_masks

