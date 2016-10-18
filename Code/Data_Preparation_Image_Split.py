%pylab inline
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.cross_validation import train_test_split
import scipy
import seaborn as sns
from spectral import imshow
from collections import Counter
import random
import pickle
pl.gray()

##loading images for input and target image
import scipy.io as io
input_image = io.loadmat('../data/Indian_pines.mat')['indian_pines']
target_image = io.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

train_image = input_image[:73]
train_labels = target_image[:73]
test_image = input_image[73:]
test_labels = target_image[73:]

#Reshaping the images to python convention
train_image = train_image.transpose((2,0,1))
train_image_height = train_image.shape[1]
train_image_width = train_image.shape[2]

print train_image_height,'x',train_image_width
print len(train_image),'x',len(train_image[0]),'x',len(train_image[0][0])

#Reshaping the images to python convention
test_image = test_image.transpose((2,0,1))
test_image_height = test_image.shape[1]
test_image_width = test_image.shape[2]

print test_image_height,'x',test_image_width
print len(test_image),'x',len(test_image[0]),'x',len(test_image[0][0])

## Scaling Down the image to 0 - 1

train_image = train_image.astype(float)
train_image -= np.min(train_image)
train_image /= np.max(train_image) - np.min(train_image) + 1

## Scaling Down the image to 0 - 1

test_image = test_image.astype(float)
test_image -= np.min(test_image)
test_image /= np.max(test_image) - np.min(test_image) + 1

## Preparing patches
def return_patches(input_image, target_image, PATCH_SIZE):
    
    input_patches = []
    targets = []
    
    input_image_height = input_image.shape[1]
    input_image_width = input_image.shape[2]



    for i in xrange(input_image_height-PATCH_SIZE+1):
        for j in xrange(input_image_width-PATCH_SIZE+1):

            height_slice = slice(i, i+PATCH_SIZE)
            width_slice = slice(j, j+PATCH_SIZE)

            patch = input_image[:, height_slice, width_slice]

            #We pick the target label as the label of the central pixel of the patch
            target = int(target_image[i+(PATCH_SIZE-1)/2, j+(PATCH_SIZE-1)/2])
            if target==0:
                continue
            else:
                targets.append(target)
                input_patches.append(patch)
                
                
    return input_patches, targets

train_patches, train_targets = return_patches(train_image, train_labels, 21)
test_patches, test_targets = return_patches(test_image, test_labels, 21)

# Neglecting all the patches belonging to class 9 and 5 in train and test image

def neglect_minority_classes(input_patches, targets, neglect_classes):
    
    patches = []
    new_targets = []
    
    for i,target in enumerate(targets):
        if target not in neglect_classes:
            patches.append(input_patches[i])
            new_targets.append(target)
    
    return patches, new_targets

train_patches, train_labels = neglect_minority_classes(train_patches, train_targets, [5, 9])
test_patches, test_labels = neglect_minority_classes(test_patches, test_targets, [5, 9])

# Oversampling Train Samples

def oversampling_patches(input_patches, targets):
    
    input_patches = np.array(input_patches)
    targets = np.array(targets)
    
    unq, unq_idx = np.unique(targets, return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    cnt = np.max(unq_cnt)
    n_targets = np.empty((cnt*len(unq),) + targets.shape[1:], targets.dtype)
    n_input_patches = np.empty((cnt*len(unq),) + input_patches.shape[1:], input_patches.dtype)
    for j in xrange(len(unq)):
        indices = np.random.choice(np.where(unq_idx==j)[0], cnt)
        n_targets[j*cnt:(j+1)*cnt] = targets[indices]
        n_input_patches[j*cnt:(j+1)*cnt] = input_patches[indices]
        
    return n_input_patches, n_targets
    

train_patches, train_labels = oversampling_patches(train_patches, train_labels)

train_zip = zip(train_patches, train_labels)
test_zip = zip(test_patches, test_labels)
random.shuffle(train_zip)
random.shuffle(test_zip)
train_patches, train_labels = zip(*train_zip)
test_patches, test_labels = zip(*test_zip)
train_patches = list(train_patches)
train_labels = list(train_labels)
test_patches = list(test_patches)
test_labels = list(test_labels)

n_test_patches = []
n_test_labels = []
n_val_patches = []
n_val_labels = []
for c in list(set(test_labels)):
	temp = []
	for i, target in enumerate(test_labels):
		if target == c:
			tempp.append(test_patches[i])
			templ.append(target)
	n_test_labels += templ[:len(templ)/2]
	n_test_patches += tempp[:len(tempp)/2]
	n_val_labels += templ[len(templ)/2:]
	n_val_patches += tempp[len(tempp)/2:]


n_train_file = len(train_patches)/500 + 1

for i in range(n_train_file):
    train_dict = {}
    if i == n_train_file:
        start = i * 500
        end = len(train_patches) + 1
    else:    
        start = i * 500
        end = (i+1) * 500
    file_name = "../data/train" + "_" + str(PATCH_SIZE) + "_" + str(i) + ".mat"
    train_dict["train_patch"] = train_patches[start:end]
    train_dict["train_labels"] = train_labels[start:end]
    io.savemat(file_name,train_dict)


n_test_file = len(n_test_patches)/500 + 1

for i in range(n_test_file):
    test_dict = {}
    if i == n_test_file:
        start = i * 500
        end = len(n_test_patches) + 1
    else:
        start = i * 500
        end = (i+1) * 500
    file_name = "../data/test" + "_" + str(PATCH_SIZE) + "_" + str(i) + ".mat"
    test_dict["test_patch"] = n_test_patches[start:end]
    test_dict["test_labels"] = n_test_labels[start:end]
    io.savemat(file_name,test_dict)

n_val_file = len(n_val_patches)/500 + 1

for i in range(n_val_file):
    val_dict = {}
    if i == n_val_file:
        start = i * 500
        end = len(n_val_patches) + 1
    else:
        start = i * 500
        end = (i+1) * 500
    file_name = "../data/test" + "_" + str(PATCH_SIZE) + "_" + str(i) + ".mat"
    test_dict["test_patch"] = n_test_patches[start:end]
    test_dict["test_labels"] = n_test_labels[start:end]
    io.savemat(file_name,test_dict)
