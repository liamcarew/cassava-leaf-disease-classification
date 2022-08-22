from tensorflow.keras.applications.densenet import DenseNet201, MobileNetV3Large
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

#function that builds deep learning model based on specified hyperparameters
def build_deep_learning_model(backbone, dropout_rate, optimiser, learning_rate, layer_num):

  #correct any input so that information used is syntactically correct
  backbone = backbone.lower()
  optimiser = optimiser.lower()

  #make sure that backbone option is inputted correctly
  assert backbone in ['densenet201', 'mobilenetv3'], 'backbone must either be \'densenet201\' or \'mobilenetv3\''

  #make sure that optimiser is inputted correctly
  assert optimiser in ['adam', 'sgd'], 'optimiser must either be \'adam\' or \'sgd\''

  #Define important global variables for training
  INPUT_SHAPE = (224,224,3)

  #define feature extractor for transfer learning
  if backbone == 'densenet201':
    backbone = DenseNet201(include_top=False,
                           weights='imagenet',
                           input_shape = INPUT_SHAPE,
                           pooling = None)
    
  elif backbone == 'mobilenetv3':
    backbone = MobileNetV3Large(include_top=False,
                                weights='imagenet',
                                input_shape = INPUT_SHAPE,
                                pooling = None)

  #define FCN to add on top of feature extractor
  dropout = Dropout(rate=dropout_rate)(backbone.output)
  global_pooling = GlobalMaxPooling2D()(dropout)
  output = Dense(units=5, activation='softmax')(global_pooling)

  #combine feature extractor and FCN to form transfer learning model
  model = models.Model(inputs = backbone.input, outputs = output)

  #freeze CNN layers up until layer from which feature maps will be extracted
  for layer in model.layers[:layer_num]:
    layer.trainable = False

  #unfreeze CNN layers from layer from which feature maps will be extracted until softmax classification layer
  for layer in model.layers[layer_num:]:
    layer.trainable = True

  #compile model
  if optimiser == 'adam':
    optimiser = Adam(learning_rate=learning_rate)
  elif optimiser == 'sgd':
    optimiser = SGD(learning_rate=learning_rate)

  loss_function = 'sparse_categorical_crossentropy'
  metric = SparseCategoricalAccuracy()

  #compile model
  model.compile(optimizer = optimiser,
                loss = loss_function,
                metrics = metric)
  
  return model
