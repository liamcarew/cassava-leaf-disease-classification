#import necessary libraries

##from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#function that returns a feature extractor based on a pre-trained CNN backbone
def build_feature_extractor(input_image_shape, cnn_backbone_name, output_layer_name, load_fine_tuned_model=False, fine_tuned_weights_path=None):

    #check that the input image size provided is (224,224,3)
    assert input_image_shape == (224,224,3), 'Input dimensions should be (224,224,3)'

    #create an input tensor for the images to go through before going into the pre-trained CNN backbone
    inputs = Input(shape=input_image_shape)

    #check that 'cnn_backbone_name' is one of the two CNN backbones you are using and that only one name is given
    assert cnn_backbone_name in ['DenseNet201', 'MobileNetV2'], 'CNN backbone should either be \'DenseNet201\', \'MobileNetV2\''

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

    #if we are using a final fine-tuned model for feature extraction, let's load the weights from this model to our model instance
    if load_fine_tuned_model:
        backbone.load_weights(fine_tuned_weights_path)
    
    #Next, let's build feature extractor using candidate layer 1 (28x28x32) as output
    cnn_feature_extractor = Model(backbone.input, backbone.get_layer(output_layer_name).output)
    
    return cnn_feature_extractor