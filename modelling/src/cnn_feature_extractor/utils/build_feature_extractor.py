#import necessary libraries

##from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#function that returns a feature extractor based on a pre-trained CNN backbone
def build_feature_extractor(input_image_shape, cnn_backbone_name, output_layer_name):

    #check that the input image size provided is (224,224,3)
    ## assert...

    #create an input tensor for the images to go through before going into the pre-trained CNN backbone
    inputs = Input(shape=input_image_shape)

    #check that 'cnn_backbone_name' is one of the two CNN backbones you are using and that only one name is given
    ## assert...

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
    
    #Next, let's build feature extractor using candidate layer 1 (28x28x32) as output
    cnn_feature_extractor = Model(backbone.input, backbone.get_layer(output_layer_name).output)
    
    return cnn_feature_extractor