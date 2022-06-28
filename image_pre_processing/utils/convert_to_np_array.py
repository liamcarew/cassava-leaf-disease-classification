import numpy as np

def convert_to_np_array(dataset, split_name):

  #get the number of images in dataset
  num_images = len(dataset)

  #created empty vectors to populate
  #x_cassava = np.empty([num_images, img_size, img_size, 3], dtype='float32')
  y_cassava = np.empty(num_images, dtype='float32')

  #populate the above vectors
  counter = 0
  for image, label in dataset: 
    #x_cassava[counter] = data["image"]
    #y_cassava[counter] = data["label"]
    #x_cassava[counter] = image
    y_cassava[counter] = label
    counter += 1

  if counter == num_images:
    print('All {} images and labels converted to NumPy arrays'.format(split_name))

  return y_cassava