"""
This py file modifies the 2D model from the standard Inception-ResNet V2 model for Keras. 

The 2D model code can be found here:
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Reference:
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)
"""

from keras import backend, layers, models, utils
import os

def conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv1D`.
        kernel_size: kernel size as in `Conv1D`.
        strides: strides in `Conv1D`.
        padding: padding mode in `Conv1D`.
        activation: activation in `Conv1D`.
        use_bias: whether to use a bias in `Conv1D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv1D` and `BatchNormalization`.
    """
    x = layers.Conv1D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 2
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv1d_bn(x, 32, 1)
        branch_1 = conv1d_bn(x, 32, 1)
        branch_1 = conv1d_bn(branch_1, 32, 3)
        branch_2 = conv1d_bn(x, 32, 1)
        branch_2 = conv1d_bn(branch_2, 48, 3)
        branch_2 = conv1d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv1d_bn(x, 192, 1)
        branch_1 = conv1d_bn(x, 128, 1)
        branch_1 = conv1d_bn(branch_1, 160, 7)
        branch_1 = conv1d_bn(branch_1, 192, 1)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv1d_bn(x, 192, 1)
        branch_1 = conv1d_bn(x, 192, 1)
        branch_1 = conv1d_bn(branch_1, 224, 3)
        branch_1 = conv1d_bn(branch_1, 256, 1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 2
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv1d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights=None,
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=100,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as input for the model.
        input_shape: optional shape tuple if input_tensor is not specified. 
            and width should be no smaller than 75.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 3D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify inputs
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization) or the path to the weights file to be loaded.')

    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    # Stem block: 35 x 192
    x = conv1d_bn(inputs, 32, 3, strides=2, padding='valid')
    x = conv1d_bn(x, 32, 3, padding='valid')
    x = conv1d_bn(x, 64, 3)
    x = layers.MaxPooling1D(3, strides=2)(x)
    x = conv1d_bn(x, 80, 1, padding='valid')
    x = conv1d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling1D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 320
    branch_0 = conv1d_bn(x, 96, 1)
    branch_1 = conv1d_bn(x, 48, 1)
    branch_1 = conv1d_bn(branch_1, 64, 5)
    branch_2 = conv1d_bn(x, 64, 1)
    branch_2 = conv1d_bn(branch_2, 96, 3)
    branch_2 = conv1d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 2
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 1088
    branch_0 = conv1d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv1d_bn(x, 256, 1)
    branch_1 = conv1d_bn(branch_1, 256, 3)
    branch_1 = conv1d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 2080
    branch_0 = conv1d_bn(x, 256, 1)
    branch_0 = conv1d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv1d_bn(x, 256, 1)
    branch_1 = conv1d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv1d_bn(x, 256, 1)
    branch_2 = conv1d_bn(branch_2, 288, 3)
    branch_2 = conv1d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 1536
    x = conv1d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = models.Model(inputs, x, name='1d_inception_resnet_v2')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model