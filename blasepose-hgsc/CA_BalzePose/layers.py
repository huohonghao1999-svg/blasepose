import tensorflow as tf

class ChannelAttention(tf.keras.layers.Layer):
    """
    Channel Attention (CA) mechanism for improving end-point keypoint detection
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.dense1 = tf.keras.layers.Dense(
            units=self.channels // self.reduction_ratio,
            activation='relu',
            use_bias=False
        )
        self.dense2 = tf.keras.layers.Dense(
            units=self.channels,
            activation='sigmoid',
            use_bias=False
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Global Average Pooling
        gap = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        
        # Channel attention weights
        attention_weights = self.dense1(gap)
        attention_weights = self.dense2(attention_weights)
        
        # Apply attention weights
        return inputs * attention_weights
        
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio
        })
        return config

class ChannelPadding(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def call(self, input):
        # Calculate padding dynamically in call method
        input_channels = tf.shape(input)[-1]
        padding_needed = self.channels - input_channels
        pad_shape = tf.stack([
            tf.constant(0), tf.constant(0),  # batch dimension
            tf.constant(0), tf.constant(0),  # height dimension  
            tf.constant(0), tf.constant(0),  # width dimension
            tf.constant(0), padding_needed   # channel dimension
        ])
        pad_shape = tf.reshape(pad_shape, [4, 2])
        return tf.pad(input, pad_shape)

class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num = 3, channel = 48, channel_padding = 1, use_ca_attention = False):
        super(BlazeBlock, self).__init__()
        self.use_ca_attention = use_ca_attention
        
        # <----- downsample ----->
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                # # 因为我实在是不会写channel padding的实现，所以这里用了个1x1的卷积来凑个数，嘤~
                # tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
                # Update: 最终，还是自己写出来了，嘤～
                ChannelPadding(channels=channel)
            ])
        else:
            # channel number invariance
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
        # <----- separable convolution ----->
        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ]))
        
        # Add Channel Attention for end-point keypoint enhancement
        if self.use_ca_attention:
            self.channel_attention = ChannelAttention(reduction_ratio=16)

    def call(self, x):
        x = tf.keras.activations.relu(self.downsample_a(x) + self.downsample_b(x))
        for i in range(len(self.conv)):
            x = tf.keras.activations.relu(x + self.conv[i](x))
        
        # Apply Channel Attention for end-point keypoint enhancement
        if self.use_ca_attention:
            x = self.channel_attention(x)
            
        return x