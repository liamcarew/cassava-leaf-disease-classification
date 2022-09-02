import numpy as np
from sklearn import utils
import tensorflow as tf
import time
import tracemalloc

#read in 'x_train' and 'y_train'

print('Loading training images and associated labels...\n')

x_train = np.load('/scratch/crwlia001/data/training_set/original/x_train.npy')
y_train = np.load('/scratch/crwlia001/data/y_train.npy')

print('training images and associated labels loaded!\n')

#function that returns the indices in NumPy array that correspond to a given class
def get_class_indices(label_array):

  index_pos_cbb = np.where(label_array == 0)[0]
  index_pos_cbsd = np.where(label_array == 1)[0]
  index_pos_cgm = np.where(label_array == 2)[0]
  index_pos_cmd = np.where(label_array == 3)[0]
  index_pos_healthy = np.where(label_array == 4)[0]

  return index_pos_cbb, index_pos_cbsd, index_pos_cgm, index_pos_cmd, index_pos_healthy

#return the class indices for each class in the training image and label arrays
index_pos_cbb, index_pos_cbsd, index_pos_cgm, index_pos_cmd, index_pos_healthy = get_class_indices(y_train)

#separate the original training arrays by class
x_train_cbb, y_train_cbb = x_train[index_pos_cbb], y_train[index_pos_cbb]
x_train_cbsd, y_train_cbsd = x_train[index_pos_cbsd], y_train[index_pos_cbsd]
x_train_cgm, y_train_cgm = x_train[index_pos_cgm], y_train[index_pos_cgm]
x_train_cmd, y_train_cmd = x_train[index_pos_cmd], y_train[index_pos_cmd]
x_train_healthy, y_train_healthy = x_train[index_pos_healthy], y_train[index_pos_healthy]

#delete 'x_train' and 'y_train'
del x_train
del y_train

#function that takes in an image and applies an augmentation
def augment_image(image, augmentation, aug_img_array, counter):

  #check that augmentation name is correct
  assert augmentation in ['vertical_flip', 'horizontal_flip', 'vertical_and_horizontal_flip']

  #vertical flip
  if augmentation == 'vertical_flip':
    augmented_img = tf.image.flip_up_down(image)
  
  #horizontal flip
  elif augmentation == 'horizontal_flip':
    augmented_img = tf.image.flip_left_right(image)

  #vertical and horizontal flip
  elif augmentation == 'vertical_and_horizontal_flip':
    augmented_img = tf.image.flip_up_down(tf.image.flip_left_right(image))

  #assign new augmented images and associated labels to position in 'aug' arrays
  aug_img_array[counter] = augmented_img
  #aug_label_array[counter] = int(np.unique(label_array))

  #increment counter
  counter += 1

#function that returns a sample of specified size
def class_sample(img_array, label, sample_size, random_state):
  
  #get a random sample of 20 image indices from each class (balanced 100 image sample)
  np.random.seed(random_state)
  aug_img_sample_inds = np.random.choice(list(range(len(img_array))), size=sample_size, replace=False)
  aug_img_sample = img_array[aug_img_sample_inds]
  #aug_label_sample = np.fill(sample_size, label)

  return aug_img_sample

#function that takes in an image and label array for a given minority class and returns balanced image and label arrays (1200) than include all the original images with the augmented ones added
def upsample_minority_class(img_array, label, no_aug_combs, balanced_sample_size):

  #specify image size in array (height, width, n_channels)
  #IMG_SIZE = (224,224,3)

  #create an empty array to populate with augmented images
  aug_img_array = np.empty([len(img_array)*no_aug_combs, 224, 224, 3], dtype='uint8')

  #also create an associated label array using the class label
  #aug_label_array = np.empty(len(img_array)*no_aug_combs, dtype='float32')

  #iterate through 'img_array' and apply augmentations using tf functions. Assign these new augmented images to positions in a new array called 'aug_img_array' (this array excludes the original images) along with an associated label array
  counter = 0
  for image in img_array:
    
    #apply first augmentation and save this in 'aug_img_array'
    augment_image(image = image,
                  augmentation = 'vertical_flip',
                  aug_img_array = aug_img_array,
                  counter = counter)

    #apply second augmentation and save this in 'aug_img_array'
    augment_image(image = image,
                  augmentation = 'horizontal_flip',
                  aug_img_array = aug_img_array,
                  counter = counter)

    #apply third augmentation and save this in 'aug_img_array'
    augment_image(image = image,              
                  augmentation = 'vertical_and_horizontal_flip',
                  aug_img_array = aug_img_array,
                  counter = counter)

  #Shuffle this array
  #aug_img_array, aug_label_array = utils.shuffle(aug_img_array, aug_label_array)

  #find the difference between the length of 'img_array' and the goal number of observation in balanced sample
  num_aug_imgs_needed = balanced_sample_size - len(img_array)

  #randomly sample the difference needed from 'aug_img_array' and the associated labels in 'aug_label_array' ('aug_sample_img_array' and 'aug_sample_label_array')
  aug_sample_img_array = class_sample(img_array = aug_img_array,
                                      label = label,
                                      sample_size = num_aug_imgs_needed,
                                      random_state = 1)
  
  #print(aug_sample_img_array.shape)
  #print(img_array.shape)

  #combine 'img_array' with 'aug_sample_img_array', and 'label_array' with 'aug_sample_label_array'
  balanced_class_after_aug_imgs = np.vstack((img_array, aug_sample_img_array))
  balanced_class_after_aug_labels = np.empty(len(balanced_class_after_aug_imgs))
  balanced_class_after_aug_labels.fill(label)

  #return balanced image and label arrays
  return balanced_class_after_aug_imgs, balanced_class_after_aug_labels

#dictionaries to add peak RAM usage and execution time to
aug_mem_usage = {}
aug_time = {}

#begin peak RAM usage and execution time measurement
tracemalloc.start()
start_time_training = time.process_time()

##upsample the minority classes with offline data augmentation

#CBB
cbb_img_balanced, cbb_labels_balanced = upsample_minority_class(img_array = x_train_cbb,
                                                                label = 0,
                                                                no_aug_combs = 3,
                                                                balanced_sample_size = 1200)

#CGM
cgm_img_balanced, cgm_labels_balanced = upsample_minority_class(img_array = x_train_cgm,
                                                                label = 2,
                                                                no_aug_combs = 3,
                                                                balanced_sample_size = 1200)

#Healthy
healthy_img_balanced, healthy_labels_balanced = upsample_minority_class(img_array = x_train_healthy,
                                                                        label = 4,
                                                                        no_aug_combs = 3,
                                                                        balanced_sample_size = 1200)

##downsample the majority classes

#CBSD
cbsd_img_balanced = class_sample(img_array = x_train_cbsd,
                                 label = 1,
                                 sample_size = 1200,
                                 random_state = 1)

cbsd_labels_balanced = np.empty(len(cbsd_img_balanced))
cbsd_labels_balanced.fill(1)

#CMD
cmd_img_balanced = class_sample(img_array = x_train_cmd,
                                label = 3,
                                sample_size = 1200,
                                random_state = 1)

cmd_labels_balanced = np.empty(len(cmd_img_balanced))
cmd_labels_balanced.fill(3)

#delete unnecessary variables
del x_train_cbb
del y_train_cbb
del x_train_cbsd
del y_train_cbsd
del x_train_cgm
del y_train_cgm
del x_train_cmd
del y_train_cmd
del x_train_healthy
del y_train_healthy

#combine separate class images and labels into unified image and label arrays
balanced_x_train = np.vstack((cbb_img_balanced, cbsd_img_balanced, cgm_img_balanced, cmd_img_balanced, healthy_img_balanced))
balanced_y_train = np.ravel(np.hstack((cbb_labels_balanced, cbsd_labels_balanced, cgm_labels_balanced, cmd_labels_balanced, healthy_labels_balanced)))

##shuffle to remove any structural bias
balanced_x_train, balanced_y_train = utils.shuffle(balanced_x_train, balanced_y_train)

#terminate monitoring of RAM usage and execution time
end_time_training = time.process_time()
first_size, first_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

#get the execution time
training_exec_time = end_time_training - start_time_training

#assign these values to respective dictionaries
aug_mem_usage['augmentation'] = first_peak / 1000000 #convert from bytes to megabytes
aug_time['augmentation'] = training_exec_time #in seconds

#check that you get the correct result
assert len(balanced_x_train) == 6000
assert len(balanced_y_train) == 6000

#save augmented training images and labels
np.save('/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy', balanced_x_train)
np.save('/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy', balanced_y_train)

#save dictionaries
np.save('/home/crwlia001/augmentation/aug_mem_usage.npy', aug_mem_usage)
np.save('/home/crwlia001/augmentation/aug_time.npy', aug_time)