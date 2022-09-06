from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

#define a wrapper function that builds the deep learning model
def build_deep_learning_model(backbone, dropout_rate, optimiser, learning_rate, start_fine_tune_layer_name):
  
  #correct any input so that information used is syntactically correct
  backbone = backbone.lower()
  optimiser = optimiser.lower()

  #make sure that backbone option is inputted correctly
  assert backbone in ['densenet201', 'mobilenetv2'], 'backbone must either be \'densenet201\' or \'mobilenetv2\''

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
    
  elif backbone == 'mobilenetv2':
    backbone = MobileNetV2(include_top=False,
                           weights='imagenet',
                           input_shape = INPUT_SHAPE,
                           pooling = None)

  #define FCN to add on top of feature extractor
  dropout = Dropout(rate=dropout_rate)(backbone.output)
  global_pooling = GlobalMaxPooling2D()(dropout)
  output = Dense(units=5, activation='softmax')(global_pooling)

  #combine feature extractor and FCN to form transfer learning model
  model = models.Model(inputs = backbone.input, outputs = output)

  #create list of layer names
  model_layer_names = []
  for layer in model.layers:
    model_layer_names.append(layer.name)

  #unfreeze certain layers in 'model' (for fine-tuning)
  for layer in model.layers[model_layer_names.index(start_fine_tune_layer_name):]:
    layer.trainable = True

  #freeze all other layers
  for layer in model.layers[:model_layer_names.index(start_fine_tune_layer_name)]:
    layer.trainable = False

  # #freeze CNN layers up until layer from which feature maps will be extracted
  # for layer in model.layers[:layer_num]:
  #   layer.trainable = False

  # #unfreeze CNN layers from layer from which feature maps will be extracted until softmax classification layer
  # for layer in model.layers[layer_num:]:
  #   layer.trainable = True

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
