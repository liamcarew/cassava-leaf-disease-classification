#import necessary libraries

##from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D

#function that returns a feature extractor based on a pre-trained CNN backbone
def build_feature_extractor(input_image_shape, cnn_backbone_name, output_layer_name, load_fine_tuned_model=False, best_dropout_rate=None, fine_tuned_weights_path=None):
    """
    Function that produces a CNN feature extractor from the transfer learning model specified

    Args:
        input_image_shape (int): The shape of the input dimensions needed for the transfer learning model
        cnn_backbone_name (str): The name of the transfer learning model to load (either 'DenseNet201' or 'MobileNetV2')
        output_layer_name (str): The name of the layer in the convolutional module from which to extract feature maps
        load_fine_tuned_model (bool, optional): Whether to load a weights vector from a pre-trained model before building the feature extractor. Defaults to False.
        best_dropout_rate (int, optional): The dropout rate of the pre-trained models from which weights are being loaded. Defaults to None.
        fine_tuned_weights_path (str, optional): the path to the pre-trained weights vector. Defaults to None.

    Returns:
        cnn_feature extractor: The CNN feature extractor built based on the parameter values. 
    """

    #check that the input image size provided is (224,224,3)
    assert input_image_shape == (224,224,3), 'Input dimensions should be (224,224,3)'

    #create an input tensor for the images to go through before going into the pre-trained CNN backbone
    inputs = Input(shape=input_image_shape)

    #check that 'cnn_backbone_name' is one of the two CNN backbones you are using, and that only one name is given
    assert cnn_backbone_name in ['DenseNet201', 'MobileNetV2'], 'CNN backbone should either be \'DenseNet201\' or \'MobileNetV2\''

    #import the pre-trained CNN backbone specified
    if cnn_backbone_name == 'DenseNet201':
        backbone = DenseNet201(
            include_top=False,
            weights='imagenet',
            input_tensor = inputs,
            input_shape = input_image_shape,
            pooling = None
            )

    elif cnn_backbone_name == 'MobileNetV2':
        backbone = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor = inputs,
            input_shape = input_image_shape,
            pooling = None
            )

    #if using a final fine-tuned model for feature extraction, load the weights from this model to our model instance
    if load_fine_tuned_model:

        #define fully-connected network (FCN) to add on top of feature extractor
        dropout = Dropout(rate=best_dropout_rate)(backbone.output)
        global_pooling = GlobalMaxPooling2D()(dropout)
        output = Dense(units=5, activation='softmax')(global_pooling)

        #convert this to a 'Model' instance so that fine-tuned weights can be loaded
        model = Model(inputs = backbone.input, outputs = output)

        #load model weights
        model.load_weights(fine_tuned_weights_path)

        #convert this into feature extractor
        cnn_feature_extractor = Model(model.input, model.get_layer(output_layer_name).output)
    
    else:

        #Build feature extractor using candidate layer 1
        cnn_feature_extractor = Model(backbone.input, backbone.get_layer(output_layer_name).output)
    
    return cnn_feature_extractor