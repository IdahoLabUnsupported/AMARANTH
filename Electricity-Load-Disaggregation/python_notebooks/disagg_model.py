###############################################################################
# Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Written by Bradley Marx 08/27/2025
#
# Keras model for PCA of power disaggregation
###############################################################################

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable()
def build_con1d_block(channels, kernels, d_rates, use_norm: bool=True, do_rt: float = 0.1, gate: bool = False):
    conv_block = tf.keras.Sequential([])
    
    for channel, kernel, d_rate in zip(channels, kernels, d_rates):
        if gate: 
            conv_block.add(layers.Conv1D(channel, kernel, dilation_rate=d_rate, padding='SAME', activation='sigmoid'))
        else:
            conv_block.add(layers.Conv1D(channel, kernel, dilation_rate=d_rate, padding='SAME'))
            if use_norm:
                conv_block.add(layers.BatchNormalization())
            conv_block.add(layers.LeakyReLU(0.05))
            
        conv_block.add(layers.Dropout(do_rt))

    return conv_block


@tf.keras.utils.register_keras_serializable()
class XPixelSCPA(tf.keras.layers.Layer):
    '''Self-Calibrated Pixel Attention Block'''
    def __init__(self, dim: int, do_rt: float, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.do_rt = do_rt

        # Define layers
        self.pregate_lyr =  build_con1d_block([self.dim // 2], [3], [1], do_rt=self.do_rt)
        self.postgate_lyr = build_con1d_block([self.dim // 2], [3], [1], do_rt=self.do_rt)
        self.skip_lyr =     build_con1d_block([self.dim // 2], [3], [1], do_rt=self.do_rt)
        
        self.inp_gate_lyr = build_con1d_block([self.dim // 2], [1], [1], do_rt=self.do_rt)
        self.inp_skip_lyr = build_con1d_block([self.dim // 2], [1], [1], do_rt=self.do_rt)
        self.gate_lyr =     build_con1d_block([self.dim // 2], [1], [1], do_rt=self.do_rt, gate=True)
        self.output_lyr =   build_con1d_block([self.dim],      [1], [1], do_rt=self.do_rt)

    def call(self, inputs, training=False):
        # Input shape: (Batch, seq_len, dims)
        
        gated_input = self.inp_gate_lyr(inputs, training=training)
        gate_vals = self.gate_lyr(gated_input, training=training)
        pregate_conv = self.pregate_lyr(gated_input, training=training)
        postgate_vals = pregate_conv * gate_vals
        postgate_conv = self.postgate_lyr(postgate_vals, training=training)

        skip_input = self.inp_skip_lyr(inputs, training=training)
        skip_conv = self.skip_lyr(skip_input, training=training)

        concat_vals = tf.concat([postgate_conv, skip_conv], axis=-1)
        concat_conv = self.output_lyr(concat_vals, training=training)

        output = inputs + concat_conv

        return output

    def get_config(self):
        return {"dim": self.dim,
                "do_rt": self.do_rt}


@tf.keras.utils.register_keras_serializable()
class XPixelUPA(tf.keras.layers.Layer):
    '''Uncalibrated Pixel Attention Block'''
    def __init__(self, dim: int, do_rt: float, out_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.do_rt = do_rt
        self.out_dim = out_dim

        # Define layers
        self.pregate_lyr =  build_con1d_block([self.dim], [3], [1], do_rt=self.do_rt)
        self.postgate_lyr = build_con1d_block([self.dim], [3], [1], do_rt=self.do_rt)
        
        self.gate_lyr =     build_con1d_block([self.dim], [1], [1], do_rt=self.do_rt, gate=True)
        if self.out_dim:
            self.postgate_lyr = tf.keras.Sequential([self.postgate_lyr,
                                                     layers.Conv1D(self.out_dim, 1, padding='SAME', activation='relu')])

    def call(self, inputs, training=False):
        # Input shape: (Batch, seq_len, dims)
        
        pregate_conv = self.pregate_lyr(inputs, training=training)
        gate_vals = self.gate_lyr(pregate_conv, training=training)
        postgate_vals = pregate_conv * gate_vals
        
        postgate_conv = self.postgate_lyr(postgate_vals, training=training)
        if not self.out_dim:
            postgate_conv = postgate_conv + inputs
        return postgate_conv

    def get_config(self):
        return {"dim": self.dim,
                "do_rt": self.do_rt,
                "out_dim": self.out_dim}
        

@tf.keras.utils.register_keras_serializable()
class PowerDisaggregator(tf.keras.Model):
    def __init__(self, window_size: int, 
                 n_targets: int, 
                 n_blocks: int = 9, 
                 dim: int = 192, 
                 do_rt: float = 0.15, 
                 out_head_depth: int = 16,
                 drop_zeros: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.win_size = window_size
        self.n_blocks = n_blocks
        self.dim = dim
        self.do_rt = do_rt
        self.n_targets = n_targets
        self.out_head_depth = out_head_depth
        self.drop_zeros = drop_zeros

        self.time_conv_blocks = []
        for _ in range(self.n_blocks):
            self.time_conv_blocks.append(XPixelSCPA(dim=self.dim, do_rt=self.do_rt))
        
        self.time_ops_skip_1 = tf.keras.Sequential(layers=[
                                                  layers.Conv1D(self.dim, 1, strides=1, padding='SAME'),
                                                  layers.BatchNormalization(),
                                                  layers.LeakyReLU(0.05),
                                                  layers.Dropout(do_rt)],  name='time_domain_skip_connection')

        self.time_conv_outs = []

        for _ in range(self.n_targets):
            output_head = tf.keras.Sequential([XPixelUPA(self.dim, self.do_rt) for _ in range(self.out_head_depth)] + [XPixelUPA(self.dim, self.do_rt, 1)])
            self.time_conv_outs.append(output_head)

    def get_config(self):
        return {"window_size": self.win_size,
                "n_targets": self.n_targets,
                "do_rt": self.do_rt,
                "out_head_depth": self.out_head_depth}

    def call(self, inputs, training=False):
        
        #### TIME DOMAIN OPERATIONS
        time_signal_input = self.time_ops_skip_1(inputs, training=training)
        
        time_signal = tf.zeros_like(time_signal_input)
        time_signal += time_signal_input

        for block in self.time_conv_blocks:
            time_signal = block(time_signal, training=training)
        # (batch, seq, channels)
        time_signal = time_signal_input + time_signal

        time_output_out = []
        for out_lyr in self.time_conv_outs:
            time_output_out.append(out_lyr(time_signal, training=training))
        time_output_out = tf.concat(time_output_out, axis=-1)
        
        time_output_out = tf.identity(time_output_out, name="time_output_out")

        return {"time_output_out": time_output_out}


@tf.keras.utils.register_keras_serializable()
def dev_from_agg(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(tf.reduce_sum(tf.abs(y_pred), axis=-1, keepdims=True), tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True)) 
    return mse


# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
#     loss={
#         "time_output_out": [tf.keras.losses.MeanSquaredError(), dev_from_agg] #tf.keras.losses.MeanSquaredError()
#     },
#     metrics={
#         "time_output_out": [tf.keras.metrics.RootMeanSquaredError()],
#     }
# )


def load_model(model_path: str, window_size: int):
    # Load pretrained model from .keras file. Make sure to have run the model definition cells above before loading this in!
    model = tf.keras.models.load_model(model_path, custom_objects={'PowerDisaggregator': PowerDisaggregator, 'window_size': window_size, 'XPixelUPA': XPixelUPA, 'XPixelSCPA': XPixelSCPA, 'dev_from_agg': dev_from_agg})
    return model
