#function to perform image resizing and pixel value normalisation
def image_preprocessing(obs):

  #normalise pixel values
  obs['image'] = tf.cast(obs['image'], tf.float32)
  obs['image'] = obs['image'] / 255

  #resize image to 224 x 224 (can change this later) <-- although, Abayomi-Alli paper showed that there was a plateau of improvement in accuracy when image resolution was >128 pixels for this dataset
  obs['image'] = tf.image.resize(obs['image'], (224, 224))

  return obs['image'], obs['label']