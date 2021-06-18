+++
author = "Skylar"
title = "Misadventures in Modifying the Output of Input() "
date = "2021-06-17"
description = ""
tags = [
    "code",
    "segmentation",
    "tensorflow"
]
categories = [
    "deep learning",
]
+++

I (re-)noticed something odd today when modifying a custom U-Net model in
TensorFlow 2. The goal was to downsample the input image and segmentation maps
on the fly once they're fed into the model during training, in order to (a)
effectively increase the receptive fields of the kernels, but mostly to (b) fit
more data per batch into a GPU that likes to run out of memory.

A simple downsampling operation is just to call
`tf.keras.layers.AveragePooling3D()` on the input tensors - that is,
do something like this:

```python
class SimpleModel:
    def __init__(self):
        pass

    def model(self):
        inpt = tf.keras.layers.Input(shape=(64, 128, 128, 2))
        inpt = tf.keras.layers.AveragePooling3D()(inpt)

        # Reduce dimensions by factor of 2
        c = tf.keras.layers.Conv3D(filters=2,
                                   kernel_size=3,
                                   padding='same')(inpt)

        c = tf.keras.layers.UpSampling3D()(c)
        model = tf.keras.models.Model(inputs=[inpt], outputs=[c])
        return model
```

Or so I thought. Imagine my surprise when I got this lovely error message:
<!-- have to do this ugly bit to get proper text wrapping :( -->
>```ValueError: Graph disconnected: cannot obtain value for tensor KerasTensor(type_spec=TensorSpec(shape=(None, 64, 128, 128, 2), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'") at layer "average_pooling3d". The following previous layers were accessed without issue: []```

What's going on? After all, if I just remove the average pooling layer and run
the convolution on `inpt`, it works just fine. Why does it error out when I
stick a different layer in the middle?

Spoiler alert: I don't know! (If you do, please
[let me know](https://github.com/StellarStorm/Deep-Learning-Notes/issues)).
But there's still a way to get this to work. If we
first assign `inpt` to a new variable, and then operate on that instead, it
works just fine

```python
class SimpleModel:
    def __init__(self):
        pass

    def model(self):
        inpt = tf.keras.layers.Input(shape=(64, 128, 128, 2))
        # Call `inpt` directly
        fixed = inpt
        fixed = tf.keras.layers.AveragePooling3D()(fixed)

        c = tf.keras.layers.Conv3D(filters=2,
                                   kernel_size=3,
                                   padding='same')(fixed)

        c = tf.keras.layers.UpSampling3D()(c)
        model = tf.keras.models.Model(inputs=[inpt], outputs=[c])
        return model
```

A few interesting things to note:

1. Both `inpt` and `fixed` seem to point to the same object. If we print the
    `id()` of the variables like this,

    ```python
    inpt = tf.keras.layers.Input(shape=(64, 128, 128, 2))
    print(id(inpt))
    fixed = inpt
    print(id(inpt), id(fixed))
    print(id(inpt) == id(fixed))
    fixed = tf.keras.layers.AveragePooling3D()(fixed)
    ```

    we get something like this

    ```code
    140222044803792
    140222044803792 140222044803792
    True
    ```

2. Reassigning `inpt` to itself, that is, replacing `fixed = inpt`
with `inpt = inpt`, does not work.

*Side note: why not just downsample the image as part of the data generator or
tf.data pipeline?* That would probably make more sense in most cases, but I'd
just wanted to do it this way. Plus, a modified version of this approach is good
for chaining multi-scale networks together, like a coarse-to-fine segmentation
network, where you probably don't want to modify the data itself.

## Appendix

Fully-working demo

```python
import numpy as np
import tensorflow as tf

feat = np.random.random_sample((1, 64, 128, 128, 2))
lab= np.ones(feat.shape)

class SimpleModel:
    def __init__(self):
        pass

    def model(self):
        inpt = tf.keras.layers.Input(shape=(64, 128, 128, 2))
        fixed = inpt
        fixed = tf.keras.layers.AveragePooling3D()(fixed)

        c = tf.keras.layers.Conv3D(filters=2,
                                   kernel_size=3,
                                   padding='same')(fixed)

        c = tf.keras.layers.UpSampling3D()(c)
        model = tf.keras.models.Model(inputs=[inpt], outputs=[c])
        return model

mod = SimpleModel().model()
mod.compile(loss='categorical_crossentropy')
mod.fit(feat, lab, epochs=4)
```
