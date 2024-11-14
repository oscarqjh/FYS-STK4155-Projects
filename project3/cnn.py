import numpy as np
from autograd import grad, elementwise_grad
from typing import Callable
from copy import copy
import sys
import math
import torch
import torch.fft

from utils import derivate, LRELU, CostCrossEntropy, sigmoid, RELU, softmax
from optimisers import Optimiser, AdamOptimiser

# global variables for index readability
INPUT_INDEX = 0
NODE_INDEX = 1
BIAS_INDEX = 1
INPUT_CHANNEL_INDEX = 1
FEATURE_MAP_INDEX = 1
HEIGHT_INDEX = 2
WIDTH_INDEX = 3
KERNEL_FEATURE_MAP_INDEX = 1
KERNEL_INPUT_CHANNEL_INDEX = 0


class Layer:
    """Abstract class for layers"""
    def __init__(self, seed):
        self.seed = seed

    def _feedforward(self):
        raise NotImplementedError

    def _backpropagate(self):
        raise NotImplementedError

    def _reset_weights(self, previous_nodes):
        raise NotImplementedError
    
class Conv2DLayer(Layer):
    """Conv2DLayer class
    Description:
    ------------
    Conv2DLayer is a class for a convolutional layer in a Convolutional Neural Network (CNN). The layer
    performs a convolution operation on the input data, which is typically an image.
    
    Parameters:
    ------------
    I   input_channels (int) number of input channels in the input data
    II  feature_maps (int) number of feature maps in the layer
    III kernel_size (tuple) size of the kernel used in the convolution operation
    IV  stride (tuple) stride of the convolution operation
    V   padding (str) padding of the convolution operation, either "same" or "valid"
    VI  activation_function (activationFunctions) activation function used in the layer
    VII seed (int) seed used for random operations
    VIII reset_weights_independently (bool) if True, the weights are reset independently, if False, the weights are reset
         based on the previous layer's output
    """

    def __init__(
            self,
            input_channels,
            feature_maps,  
            kernel_size,
            stride,
            padding="same",
            activation_function: Callable = LRELU,
            seed=42,
            reset_weights_independently=True,
    ):
        super().__init__(seed)
        self.input_channels = input_channels
        self.feature_maps = feature_maps
        self.kernel_height, self.kernel_width = kernel_size
        self.v_stride, self.h_stride = stride
        self.pad = padding
        self.act_func = activation_function

        # such that the layer can be used on its own
        # outside of the CNN module
        if reset_weights_independently == True:
            self._reset_weights_independently()

    def _padding(self, X_batch, batch_type="image"):
        # note that the shape of X_batch = [inputs, input_maps, img_height, img_width]
        # same padding for images
        if self.pad == "same" and batch_type == "image":
            padded_height = X_batch.shape[HEIGHT_INDEX] + (self.kernel_height // 2) * 2
            padded_width = X_batch.shape[WIDTH_INDEX] + (self.kernel_width // 2) * 2
            half_kernel_height = self.kernel_height // 2
            half_kernel_width = self.kernel_width // 2

            # initialize padded array
            X_batch_padded = np.ndarray(
                (
                    X_batch.shape[INPUT_INDEX],
                    X_batch.shape[FEATURE_MAP_INDEX],
                    padded_height,
                    padded_width,
                )
            )

            # zero pad all images in X_batch
            for img in range(X_batch.shape[INPUT_INDEX]):
                padded_img = np.zeros(
                    (X_batch.shape[FEATURE_MAP_INDEX], padded_height, padded_width)
                )
                padded_img[
                    :,
                    half_kernel_height : padded_height - half_kernel_height,
                    half_kernel_width : padded_width - half_kernel_width,
                ] = X_batch[img, :, :, :]
                X_batch_padded[img, :, :, :] = padded_img[:, :, :]

            return X_batch_padded

        # same padding for gradients
        elif self.pad == "same" and batch_type == "grad":
            padded_height = X_batch.shape[HEIGHT_INDEX] + (self.kernel_height // 2) * 2
            padded_width = X_batch.shape[WIDTH_INDEX] + (self.kernel_width // 2) * 2
            half_kernel_height = self.kernel_height // 2
            half_kernel_width = self.kernel_width // 2

            # initialize padded array
            delta_term_padded = np.zeros(
                (
                    X_batch.shape[INPUT_INDEX],
                    X_batch.shape[FEATURE_MAP_INDEX],
                    padded_height,
                    padded_width,
                )
            )

            # zero pad delta term
            delta_term_padded[
                :, :, : X_batch.shape[HEIGHT_INDEX], : X_batch.shape[WIDTH_INDEX]
            ] = X_batch[:, :, :, :]

            return delta_term_padded

        else:
            return X_batch
        
    def _feedforward(self, X_batch):
        # note that the shape of X_batch = [inputs, input_maps, img_height, img_width]

        # pad the input batch
        X_batch_padded = self._padding(X_batch)

        # calculate height_index and width_index after stride
        strided_height = int(np.ceil(X_batch.shape[HEIGHT_INDEX] / self.v_stride))
        strided_width = int(np.ceil(X_batch.shape[WIDTH_INDEX] / self.h_stride))

        # create output array
        output = np.ndarray(
            (
                X_batch.shape[INPUT_INDEX],
                self.feature_maps,
                strided_height,
                strided_width,
            )
        )

        # save input and output for backpropagation
        self.X_batch_feedforward = X_batch
        self.output_shape = output.shape

        # checking for errors, no need to look here :)
        self._check_for_errors()

        # convolve input with kernel
        for img in range(X_batch.shape[INPUT_INDEX]):
            for chin in range(self.input_channels):
                for fmap in range(self.feature_maps):
                    out_h = 0
                    for h in range(0, X_batch.shape[HEIGHT_INDEX], self.v_stride):
                        out_w = 0
                        for w in range(0, X_batch.shape[WIDTH_INDEX], self.h_stride):
                            output[img, fmap, out_h, out_w] = np.sum(
                                X_batch_padded[
                                    img,
                                    chin,
                                    h : h + self.kernel_height,
                                    w : w + self.kernel_width,
                                ]
                                * self.kernel[chin, fmap, :, :]
                            )
                            out_w += 1
                        out_h += 1

        # Pay attention to the fact that we're not rotating the kernel by 180 degrees when filtering the image in
        # the convolutional layer, as convolution in terms of Machine Learning is a procedure known as cross-correlation
        # in image processing and signal processing

        # return a
        return self.act_func(output / (self.kernel_height))
    
    def _backpropagate(self, delta_term_next):
        # intiate matrices
        delta_term = np.zeros((self.X_batch_feedforward.shape))
        gradient_kernel = np.zeros((self.kernel.shape))

        # pad input for convolution
        X_batch_padded = self._padding(self.X_batch_feedforward)

        # Since an activation function is used at the output of the convolution layer, its derivative
        # has to be accounted for in the backpropagation -> as if ReLU was a layer on its own.
        act_derivative = derivate(self.act_func)
        delta_term_next = act_derivative(delta_term_next)

        # fill in 0's for values removed by vertical stride in feedforward
        if self.v_stride > 1:
            v_ind = 1
            for i in range(delta_term_next.shape[HEIGHT_INDEX]):
                for j in range(self.v_stride - 1):
                    delta_term_next = np.insert(
                        delta_term_next, v_ind, 0, axis=HEIGHT_INDEX
                    )
                v_ind += self.v_stride

        # fill in 0's for values removed by horizontal stride in feedforward
        if self.h_stride > 1:
            h_ind = 1
            for i in range(delta_term_next.shape[WIDTH_INDEX]):
                for k in range(self.h_stride - 1):
                    delta_term_next = np.insert(
                        delta_term_next, h_ind, 0, axis=WIDTH_INDEX
                    )
                h_ind += self.h_stride

        # crops out 0-rows and 0-columns
        delta_term_next = delta_term_next[
            :,
            :,
            : self.X_batch_feedforward.shape[HEIGHT_INDEX],
            : self.X_batch_feedforward.shape[WIDTH_INDEX],
        ]

        # the gradient received from the next layer also needs to be padded
        delta_term_next = self._padding(delta_term_next)

        # calculate delta term by convolving next delta term with kernel
        for img in range(self.X_batch_feedforward.shape[INPUT_INDEX]):
            for chin in range(self.input_channels):
                for fmap in range(self.feature_maps):
                    for h in range(self.X_batch_feedforward.shape[HEIGHT_INDEX]):
                        for w in range(self.X_batch_feedforward.shape[WIDTH_INDEX]):
                            delta_term[img, chin, h, w] = np.sum(
                                delta_term_next[
                                    img,
                                    fmap,
                                    h : h + self.kernel_height,
                                    w : w + self.kernel_width,
                                ]
                                * np.rot90(np.rot90(self.kernel[chin, fmap, :, :]))
                            )

        # calculate gradient for kernel for weight update
        # also via convolution
        for chin in range(self.input_channels):
            for fmap in range(self.feature_maps):
                for k_x in range(self.kernel_height):
                    for k_y in range(self.kernel_width):
                        gradient_kernel[chin, fmap, k_x, k_y] = np.sum(
                            X_batch_padded[
                                img,
                                chin,
                                h : h + self.kernel_height,
                                w : w + self.kernel_width,
                            ]
                            * delta_term_next[
                                img,
                                fmap,
                                h : h + self.kernel_height,
                                w : w + self.kernel_width,
                            ]
                        )
        # all kernels are updated with weight gradient of kernel
        self.kernel -= gradient_kernel

        # return delta term
        return delta_term
    
    def _reset_weights_independently(self):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # initializes kernel matrix
        self.kernel = np.ndarray(
            (
                self.input_channels,
                self.feature_maps,
                self.kernel_height,
                self.kernel_width,
            )
        )

        # randomly initializes weights
        for chin in range(self.kernel.shape[KERNEL_INPUT_CHANNEL_INDEX]):
            for fmap in range(self.kernel.shape[KERNEL_FEATURE_MAP_INDEX]):
                self.kernel[chin, fmap, :, :] = np.random.rand(
                    self.kernel_height, self.kernel_width
                )

    def _reset_weights(self, previous_nodes):
        # sets weights
        self._reset_weights_independently()

        # returns shape of output used for subsequent layer's weight initiation
        strided_height = int(
            np.ceil(previous_nodes.shape[HEIGHT_INDEX] / self.v_stride)
        )
        strided_width = int(np.ceil(previous_nodes.shape[WIDTH_INDEX] / self.h_stride))
        next_nodes = np.ones(
            (
                previous_nodes.shape[INPUT_INDEX],
                self.feature_maps,
                strided_height,
                strided_width,
            )
        )
        return next_nodes / self.kernel_height

    def _check_for_errors(self):
        if self.X_batch_feedforward.shape[INPUT_CHANNEL_INDEX] != self.input_channels:
            raise AssertionError(
                f"ERROR: Number of input channels in data ({self.X_batch_feedforward.shape[INPUT_CHANNEL_INDEX]}) is not equal to input channels in Convolution2DLayerOPT ({self.input_channels})! Please change the number of input channels of the Convolution2DLayer such that they are equal"
            )

class Conv2DLayerOPT(Conv2DLayer):
    """
    Am optimized version of the convolution layer above which
    utilizes an approach of extracting windows of size equivalent
    in size to the filter. The convoution is then performed on those
    windows instead of a full feature map.
    """

    def __init__(
        self,
        input_channels,
        feature_maps,
        kernel_size,
        stride,
        padding="same",
        act_func: Callable = LRELU,
        seed=42,
        reset_weights_independently=True,
    ):
        super().__init__(
            input_channels,
            feature_maps,
            kernel_size,
            stride,
            padding,
            act_func,
            seed,
        )
        # true if layer is used outside of CNN
        if reset_weights_independently == True:
            self._reset_weights_independently()

    def _feedforward(self, X_batch):
        # The optimized _feedforward method is difficult to understand but computationally more efficient
        # for a more "by the book" approach, please look at the _feedforward method of Convolution2DLayer

        # save the input for backpropagation
        self.X_batch_feedforward = X_batch

        # check that there are the correct amount of input channels
        self._check_for_errors()

        # calculate new shape after stride
        strided_height = int(np.ceil(X_batch.shape[HEIGHT_INDEX] / self.v_stride))
        strided_width = int(np.ceil(X_batch.shape[WIDTH_INDEX] / self.h_stride))

        # get windows of the image for more computationally efficient convolution
        # the idea is that we want to align the dimensions that we wish to matrix
        # multiply, then use a simple matrix multiplication instead of convolution.
        # then, we reshape the size back to its intended shape
        windows = self._extract_windows(X_batch)
        windows = windows.transpose(1, 0, 2, 3, 4).reshape(
            X_batch.shape[INPUT_INDEX],
            strided_height * strided_width,
            -1,
        )

        # reshape the kernel for more computationally efficient convolution
        kernel = self.kernel
        kernel = kernel.transpose(0, 2, 3, 1).reshape(
            kernel.shape[KERNEL_INPUT_CHANNEL_INDEX]
            * kernel.shape[HEIGHT_INDEX]
            * kernel.shape[WIDTH_INDEX],
            -1,
        )

        # use simple matrix calculation to obtain output
        output = (
            (windows @ kernel)
            .reshape(
                X_batch.shape[INPUT_INDEX],
                strided_height,
                strided_width,
                -1,
            )
            .transpose(0, 3, 1, 2)
        )

        # The output is reshaped and rearranged to appropriate shape
        return self.act_func(
            output / (self.kernel_height * X_batch.shape[FEATURE_MAP_INDEX])
        )

    def _backpropagate(self, delta_term_next):
        # The optimized _backpropagate method is difficult to understand but computationally more efficient
        # for a more "by the book" approach, please look at the _backpropagate method of Convolution2DLayer
        act_derivative = derivate(self.act_func)
        delta_term_next = act_derivative(delta_term_next)

        # calculate strided dimensions
        strided_height = int(
            np.ceil(self.X_batch_feedforward.shape[HEIGHT_INDEX] / self.v_stride)
        )
        strided_width = int(
            np.ceil(self.X_batch_feedforward.shape[WIDTH_INDEX] / self.h_stride)
        )

        # copy kernel
        kernel = self.kernel

        # get windows, reshape for matrix multiplication
        windows = self._extract_windows(self.X_batch_feedforward, "image").reshape(
            self.X_batch_feedforward.shape[INPUT_INDEX]
            * strided_height
            * strided_width,
            -1,
        )

        # initialize output gradient, reshape and transpose into correct shape
        # for matrix multiplication
        output_grad_tr = delta_term_next.transpose(0, 2, 3, 1).reshape(
            self.X_batch_feedforward.shape[INPUT_INDEX]
            * strided_height
            * strided_width,
            -1,
        )

        # calculate gradient kernel via simple matrix multiplication and reshaping
        gradient_kernel = (
            (windows.T @ output_grad_tr)
            .reshape(
                kernel.shape[KERNEL_INPUT_CHANNEL_INDEX],
                kernel.shape[HEIGHT_INDEX],
                kernel.shape[WIDTH_INDEX],
                kernel.shape[KERNEL_FEATURE_MAP_INDEX],
            )
            .transpose(0, 3, 1, 2)
        )

        # for computing the input gradient
        windows_out, upsampled_height, upsampled_width = self._extract_windows(
            delta_term_next, "grad"
        )

        # calculate new window dimensions
        new_windows_first_dim = (
            self.X_batch_feedforward.shape[INPUT_INDEX]
            * upsampled_height
            * upsampled_width
        )
        # ceil allows for various asymmetric kernels
        new_windows_sec_dim = int(np.ceil(windows_out.size / new_windows_first_dim))

        # reshape for matrix multiplication
        windows_out = windows_out.transpose(1, 0, 2, 3, 4).reshape(
            new_windows_first_dim, new_windows_sec_dim
        )

        # reshape for matrix multiplication
        kernel_reshaped = kernel.reshape(self.input_channels, -1)

        # calculating input gradient for next convolutional layer
        input_grad = (windows_out @ kernel_reshaped.T).reshape(
            self.X_batch_feedforward.shape[INPUT_INDEX],
            upsampled_height,
            upsampled_width,
            kernel.shape[KERNEL_INPUT_CHANNEL_INDEX],
        )
        input_grad = input_grad.transpose(0, 3, 1, 2)

        # Update the weights in the kernel
        self.kernel -= gradient_kernel

        # Output the gradient to propagate backwards
        return input_grad

    def _extract_windows(self, X_batch, batch_type="image"):
        """
        Receives as input the X_batch with shape (inputs, feature_maps, image_height, image_width)
        and extract windows of size kernel_height * kernel_width for every image and every feature_map.
        It then returns an np.ndarray of shape (image_height * image_width, inputs, feature_maps, kernel_height, kernel_width)
        which will be used either to filter the images in feedforward or to calculate the gradient.
        """

        # initialize list of windows
        windows = []

        if batch_type == "image":
            # pad the images
            X_batch_padded = self._padding(X_batch, batch_type="image")
            img_height, img_width = X_batch_padded.shape[2:]
            # For each location in the image...
            for h in range(
                0,
                X_batch.shape[HEIGHT_INDEX],
                self.v_stride,
            ):
                for w in range(
                    0,
                    X_batch.shape[WIDTH_INDEX],
                    self.h_stride,
                ):
                    # ...obtain an image patch of the original size (strided)

                    # get window
                    window = X_batch_padded[
                        :,
                        :,
                        h : h + self.kernel_height,
                        w : w + self.kernel_width,
                    ]

                    # append to list of windows
                    windows.append(window)

            # return numpy array instead of list
            return np.stack(windows)

        # In order to be able to perform backprogagation by the method of window extraction,
        # here is a modified approach to extracting the windows which allow for the necessary
        # upsampling of the gradient in case the on of the stride parameters is larger than one.

        if batch_type == "grad":

            # In the case of one of the stride parameters being odd, we have to take some
            # extra care in calculating the upsampled size of X_batch. We solve this
            # by simply flooring the result of dividing stride by 2.
            if self.v_stride < 2 or self.v_stride % 2 == 0:
                v_stride = 0
            else:
                v_stride = int(np.floor(self.v_stride / 2))

            if self.h_stride < 2 or self.h_stride % 2 == 0:
                h_stride = 0
            else:
                h_stride = int(np.floor(self.h_stride / 2))

            upsampled_height = (X_batch.shape[HEIGHT_INDEX] * self.v_stride) - v_stride
            upsampled_width = (X_batch.shape[WIDTH_INDEX] * self.h_stride) - h_stride

            # When upsampling, we need to insert rows and columns filled with zeros
            # into each feature map. How many of those we have to insert is purely
            # dependant on the value of stride parameter in the vertical and horizontal
            # direction.
            if self.v_stride > 1:
                v_ind = 1
                for i in range(X_batch.shape[HEIGHT_INDEX]):
                    for j in range(self.v_stride - 1):
                        X_batch = np.insert(X_batch, v_ind, 0, axis=HEIGHT_INDEX)
                    v_ind += self.v_stride

            if self.h_stride > 1:
                h_ind = 1
                for i in range(X_batch.shape[WIDTH_INDEX]):
                    for k in range(self.h_stride - 1):
                        X_batch = np.insert(X_batch, h_ind, 0, axis=WIDTH_INDEX)
                    h_ind += self.h_stride

            # Since the insertion of zero-filled rows and columns isn't perfect, we have
            # to assure that the resulting feature maps will have the expected upsampled height
            # and width by cutting them og at desired dimensions.

            X_batch = X_batch[:, :, :upsampled_height, :upsampled_width]

            X_batch_padded = self._padding(X_batch, batch_type="grad")

            # initialize list of windows
            windows = []

            # For each location in the image...
            for h in range(
                0,
                X_batch.shape[HEIGHT_INDEX],
                self.v_stride,
            ):
                for w in range(
                    0,
                    X_batch.shape[WIDTH_INDEX],
                    self.h_stride,
                ):
                    # ...obtain an image patch of the original size (strided)

                    # get window
                    window = X_batch_padded[
                        :, :, h : h + self.kernel_height, w : w + self.kernel_width
                    ]

                    # append window to list
                    windows.append(window)

            # return numpy array, unsampled dimensions
            return np.stack(windows), upsampled_height, upsampled_width

    def _check_for_errors(self):
        # compares input channels of data to input channels of Convolution2DLayer
        if self.X_batch_feedforward.shape[INPUT_CHANNEL_INDEX] != self.input_channels:
            raise AssertionError(
                f"ERROR: Number of input channels in data ({self.X_batch_feedforward.shape[INPUT_CHANNEL_INDEX]}) is not equal to input channels in Convolution2DLayerOPT ({self.input_channels})! Please change the number of input channels of the Convolution2DLayer such that they are equal"
            )

class Pooling2DLayer(Layer):
    def __init__(
        self,
        kernel_size,
        stride,
        seed=42,
    ):
        super().__init__(seed)
        self.kernel_height, self.kernel_width = kernel_size
        self.v_stride, self.h_stride = stride

    def _choose_pooling(self):
        raise NotImplementedError
    
    def _backpropagate_update(self):
        raise NotImplementedError

    def _feedforward(self, X_batch):
        # Saving the input for use in the backwardpass
        self.X_batch_feedforward = X_batch

        # check if user is silly
        self._check_for_errors()

        # Computing the size of the feature maps based on kernel size and the stride parameter
        strided_height = (
            X_batch.shape[HEIGHT_INDEX] - self.kernel_height
        ) // self.v_stride + 1
        if X_batch.shape[HEIGHT_INDEX] == X_batch.shape[WIDTH_INDEX]:
            strided_width = strided_height
        else:
            strided_width = (
                X_batch.shape[WIDTH_INDEX] - self.kernel_width
            ) // self.h_stride + 1

        # initialize output array
        output = np.ndarray(
            (
                X_batch.shape[INPUT_INDEX],
                X_batch.shape[FEATURE_MAP_INDEX],
                strided_height,
                strided_width,
            )
        )

        # select pooling action, either max or average pooling
        self.pooling_action = self._choose_pooling()

        # pool based on kernel size and stride
        for img in range(output.shape[INPUT_INDEX]):
            for fmap in range(output.shape[FEATURE_MAP_INDEX]):
                for h in range(strided_height):
                    for w in range(strided_width):
                        output[img, fmap, h, w] = self.pooling_action(
                            X_batch[
                                img,
                                fmap,
                                (h * self.v_stride) : (h * self.v_stride)
                                + self.kernel_height,
                                (w * self.h_stride) : (w * self.h_stride)
                                + self.kernel_width,
                            ]
                        )

        # output for feedforward in next layer
        return output

    def _backpropagate(self, delta_term_next):
        # initiate delta term array
        delta_term = np.zeros((self.X_batch_feedforward.shape))

        for img in range(delta_term_next.shape[INPUT_INDEX]):
            for fmap in range(delta_term_next.shape[FEATURE_MAP_INDEX]):
                for h in range(0, delta_term_next.shape[HEIGHT_INDEX], self.v_stride):
                    for w in range(
                        0, delta_term_next.shape[WIDTH_INDEX], self.h_stride
                    ):
                        delta_term = self._backpropagate_update(
                            img, fmap, h, w, delta_term, delta_term_next
                        )
        # returns input to backpropagation in previous layer
        return delta_term

    def _reset_weights(self, previous_nodes):
        # calculate strided height, strided width
        strided_height = (
            previous_nodes.shape[HEIGHT_INDEX] - self.kernel_height
        ) // self.v_stride + 1
        if previous_nodes.shape[HEIGHT_INDEX] == previous_nodes.shape[WIDTH_INDEX]:
            strided_width = strided_height
        else:
            strided_width = (
                previous_nodes.shape[WIDTH_INDEX] - self.kernel_width
            ) // self.h_stride + 1

        # initiate output array
        output = np.ones(
            (
                previous_nodes.shape[INPUT_INDEX],
                previous_nodes.shape[FEATURE_MAP_INDEX],
                strided_height,
                strided_width,
            )
        )

        # returns output with shape used for reset weights in next layer
        return output

    def _check_for_errors(self):
        # check if input is smaller than kernel size -> error
        assert (
            self.X_batch_feedforward.shape[WIDTH_INDEX] >= self.kernel_width
        ), f"ERROR: Pooling kernel width_index ({self.kernel_width}) larger than data width_index ({self.X_batch_feedforward.input.shape[2]}), please lower the kernel width_index of the Pooling2DLayer"
        assert (
            self.X_batch_feedforward.shape[HEIGHT_INDEX] >= self.kernel_height
        ), f"ERROR: Pooling kernel height_index ({self.kernel_height}) larger than data height_index ({self.X_batch_feedforward.input.shape[3]}), please lower the kernel height_index of the Pooling2DLayer"

class MaxPooling2DLayer(Pooling2DLayer):
    def __init__(
        self,
        kernel_size,
        stride,
        seed=42,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            seed=seed,
        )

    def _choose_pooling(self):
        return np.max
    
    def _backpropagate_update(self, img, fmap, h, w, delta_term, delta_term_next):
        # get window
        window = self.X_batch_feedforward[
            img,
            fmap,
            h : h + self.kernel_height,
            w : w + self.kernel_width,
        ]

        # find max values indices in window
        max_h, max_w = np.unravel_index(
            window.argmax(), window.shape
        )

        # set values in new, upsampled delta term
        delta_term[
            img,
            fmap,
            (h + max_h),
            (w + max_w),
        ] += delta_term_next[img, fmap, h, w]

        return delta_term

class AveragePooling2DLayer(Pooling2DLayer):
    def __init__(
        self,
        kernel_size,
        stride,
        seed=42,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            seed=seed,
        )

    def _choose_pooling(self):
        return np.mean
    
    def _backpropagate_update(self, img, fmap, h, w, delta_term, delta_term_next):
        delta_term[
            img,
            fmap,
            h : h + self.kernel_height,
            w : w + self.kernel_width,
        ] = (
            delta_term_next[img, fmap, h, w]
            / self.kernel_height
            / self.kernel_width
        )

        return delta_term
    
class FlattenLayer(Layer):
    def __init__(self, act_func=LRELU, seed=None):
        super().__init__(seed)
        self.act_func = act_func

    def _feedforward(self, X_batch):
        # save input for backpropagation
        self.X_batch_feedforward_shape = X_batch.shape
        # Remember, the data has the following shape: (I, FM, H, W, ) in the convolutional layers
        # whilst the data has the shape (I, FM * H * W) in the fully connected layers
        # I = Inputs, FM = Feature Maps, H = Height and W = Width.
        X_batch = X_batch.reshape(
            X_batch.shape[INPUT_INDEX],
            X_batch.shape[FEATURE_MAP_INDEX]
            * X_batch.shape[HEIGHT_INDEX]
            * X_batch.shape[WIDTH_INDEX],
        )

        # add bias to a
        self.z_matrix = X_batch
        bias = np.ones((X_batch.shape[INPUT_INDEX], 1)) * 0.01
        self.a_matrix = np.hstack([bias, X_batch])

        # return a, the input to feedforward in next layer
        return self.a_matrix

    def _backpropagate(self, weights_next, delta_term_next):
        activation_derivative = derivate(self.act_func)

        # calculate delta term
        delta_term = (
            weights_next[BIAS_INDEX:, :] @ delta_term_next.T
        ).T * activation_derivative(self.z_matrix)

        # FlattenLayer does not update weights
        # reshapes delta layer to convolutional layer data format [Input, Feature_Maps, Height, Width]
        return delta_term.reshape(self.X_batch_feedforward_shape)

    def _reset_weights(self, previous_nodes):
        # note that the previous nodes to the FlattenLayer are from the convolutional layers
        previous_nodes = previous_nodes.reshape(
            previous_nodes.shape[INPUT_INDEX],
            previous_nodes.shape[FEATURE_MAP_INDEX]
            * previous_nodes.shape[HEIGHT_INDEX]
            * previous_nodes.shape[WIDTH_INDEX],
        )

        # return shape used in reset_weights in next layer
        return previous_nodes.shape[NODE_INDEX]

    def get_prev_a(self):
        return self.a_matrix
    

class FullyConnectedLayer(Layer):
    # FullyConnectedLayer per default uses LRELU and Adam Optimiser
    # with an eta of 0.0001, rho of 0.9 and rho2 of 0.999
    def __init__(
        self,
        nodes: int,
        act_func: Callable = LRELU,
        optimiser: Optimiser = AdamOptimiser(learning_rate=1e-4, beta1=0.9, beta2=0.999),
        seed: int = 42,
    ):
        super().__init__(seed)
        self.nodes = nodes
        self.act_func = act_func
        self.optimiser_weight = copy(optimiser)
        self.optimiser_bias = copy(optimiser)

        # initiate matrices for later
        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

    def _feedforward(self, X_batch):
        # calculate z
        self.z_matrix = X_batch @ self.weights

        # calculate a, add bias
        bias = np.ones((X_batch.shape[INPUT_INDEX], 1)) * 0.01
        self.a_matrix = self.act_func(self.z_matrix)
        self.a_matrix = np.hstack([bias, self.a_matrix])

        # return a, the input for feedforward in next layer
        return self.a_matrix

    def _backpropagate(self, weights_next, delta_term_next, a_previous, lam):
        # take the derivative of the activation function
        activation_derivative = derivate(self.act_func)

        # calculate the delta term
        delta_term = (
            weights_next[BIAS_INDEX:, :] @ delta_term_next.T
        ).T * activation_derivative(self.z_matrix)

        # intitiate matrix to store gradient
        # note that we exclude the bias term, which we will calculate later
        gradient_weights = np.zeros(
            (
                a_previous.shape[INPUT_INDEX],
                a_previous.shape[NODE_INDEX] - BIAS_INDEX,
                delta_term.shape[NODE_INDEX],
            )
        )

        # calculate gradient = delta term * previous a
        for i in range(len(delta_term)):
            gradient_weights[i, :, :] = np.outer(
                a_previous[i, BIAS_INDEX:], delta_term[i, :]
            )

        # sum the gradient, divide by input_index
        gradient_weights = np.mean(gradient_weights, axis=INPUT_INDEX)
        # for the bias gradient we do not multiply by previous a
        gradient_bias = np.mean(delta_term, axis=INPUT_INDEX).reshape(
            1, delta_term.shape[NODE_INDEX]
        )

        # regularization term
        gradient_weights += self.weights[BIAS_INDEX:, :] * lam

        # send gradients into optimiser
        # returns update matrix which will be used to update the weights and bias
        update_matrix = np.vstack(
            [
                self.optimiser_bias.update_change(gradient_bias),
                self.optimiser_weight.update_change(gradient_weights),
            ]
        )

        # update weights
        self.weights -= update_matrix

        # return weights and delta term, input for backpropagation in previous layer
        return self.weights, delta_term

    def _reset_weights(self, previous_nodes):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # add bias, initiate random weights
        bias = 1
        self.weights = np.random.randn(previous_nodes + bias, self.nodes)

        # returns number of nodes, used for reset_weights in next layer
        return self.nodes

    def _reset_optimiser(self):
        # resets optimiser per epoch
        self.optimiser_weight.reset()
        self.optimiser_bias.reset()

    def get_prev_a(self):
        # returns a matrix, used in backpropagation
        return self.a_matrix


class OutputLayer(FullyConnectedLayer):
    def __init__(
        self,
        nodes: int,
        output_func: Callable = LRELU,
        cost_func: Callable = CostCrossEntropy,
        optimiser: Optimiser = AdamOptimiser(learning_rate=1e-4, beta1=0.9, beta2=0.999),
        seed: int = 42,
    ):
        super().__init__(nodes, output_func, copy(optimiser), seed)
        self.cost_func = cost_func

        # initiate matrices for later
        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

        # decides if the output layer performs binary or multi-class classification
        self._set_pred_format()

    def _feedforward(self, X_batch: np.ndarray):
        # calculate a, z
        # note that bias is not added as this would create an extra output class
        self.z_matrix = X_batch @ self.weights
        self.a_matrix = self.act_func(self.z_matrix)

        # returns prediction
        return self.a_matrix

    def _backpropagate(self, target, a_previous, lam):
        # note that in the OutputLayer the activation function is the output function
        activation_derivative = derivate(self.act_func)

        # calculate output delta terms
        # for multi-class or binary classification
        if self.pred_format == "Multi-class":
            delta_term = self.a_matrix - target
        else:
            cost_func_derivative = grad(self.cost_func(target))
            delta_term = activation_derivative(self.z_matrix) * cost_func_derivative(
                self.a_matrix
            )

        # intiate matrix that stores gradient
        gradient_weights = np.zeros(
            (
                a_previous.shape[INPUT_INDEX],
                a_previous.shape[NODE_INDEX] - BIAS_INDEX,
                delta_term.shape[NODE_INDEX],
            )
        )

        # calculate gradient = delta term * previous a
        for i in range(len(delta_term)):
            gradient_weights[i, :, :] = np.outer(
                a_previous[i, BIAS_INDEX:], delta_term[i, :]
            )

        # sum the gradient, divide by input_index
        gradient_weights = np.mean(gradient_weights, axis=INPUT_INDEX)
        # for the bias gradient we do not multiply by previous a
        gradient_bias = np.mean(delta_term, axis=INPUT_INDEX).reshape(
            1, delta_term.shape[NODE_INDEX]
        )

        # regularization term
        gradient_weights += self.weights[BIAS_INDEX:, :] * lam

        # send gradients into optimiser
        # returns update matrix which will be used to update the weights and bias
        update_matrix = np.vstack(
            [
                self.optimiser_bias.update_change(gradient_bias),
                self.optimiser_weight.update_change(gradient_weights),
            ]
        )

        # update weights
        self.weights -= update_matrix

        # return weights and delta term, input for backpropagation in previous layer
        return self.weights, delta_term

    def _reset_weights(self, previous_nodes):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # add bias, initiate random weights
        bias = 1
        self.weights = np.random.rand(previous_nodes + bias, self.nodes)

        # returns number of nodes, used for reset_weights in next layer
        return self.nodes

    def _reset_optimiser(self):
        # resets optimiser per epoch
        self.optimiser_weight.reset()
        self.optimiser_bias.reset()

    def _set_pred_format(self):
        # sets prediction format to either regression, binary or multi-class classification
        if self.act_func.__name__ is None or self.act_func.__name__ == "identity":
            self.pred_format = "Regression"
        elif self.act_func.__name__ == "sigmoid" or self.act_func.__name__ == "tanh":
            self.pred_format = "Binary"
        else:
            self.pred_format = "Multi-class"

    def get_pred_format(self):
        # returns format of prediction
        return self.pred_format
    
class CNN:
    def __init__(
        self,
        cost_func: Callable = CostCrossEntropy,
        optimiser: Optimiser = AdamOptimiser(learning_rate=1e-4, beta1=0.9, beta2=0.999),
        seed: int = None,
    ):
        """
        Description:
        ------------
            Instantiates CNN object

        Parameters:
        ------------
            I   output_func (costFunctions) cost function for feed forward neural network part of CNN,
                such as "CostLogReg", "CostOLS" or "CostCrossEntropy"

            II  optimiser (Optimiser) optional parameter, default set to Adam. Can also be set to other
                optimisers such as AdaGrad, Momentum, RMS_prop and Basic. Note that optimisers have
                to be instantiated first with proper parameters (for example learning_rate, beta1 and beta2 for Adam)

            III seed (int) used for seeding all random operations
        """
        self.layers = list()
        self.cost_func = cost_func
        self.optimiser = optimiser
        self.optimiser_weight = list()
        self.optimiser_bias = list()
        self.seed = seed
        self.pred_format = None
    
    def add(self, layer):
        """
        Description:
        ------------
            Add a layer to the CNN

        Parameters:
        ------------
            I   layer (Layer) layer to be added to the CNN
        """
        # assert fully connected layer is added after flatten layer
        if isinstance(layer, FullyConnectedLayer):
            assert isinstance(self.layers[-1], FlattenLayer) or isinstance(self.layers[-1], FullyConnectedLayer), "FullyConnectedLayer should follow FlattenLayer or another FullyConnectedLayer in CNN"
            layer.optimiser_weight = copy(self.optimiser)
            layer.optimiser_bias = copy(self.optimiser)
        # assert output layer is added after fully connected layer
        if isinstance(layer, OutputLayer):
            assert isinstance(self.layers[-1], FullyConnectedLayer), "OutputLayer should follow FullyConnectedLayer in CNN"
            self.pred_format = layer.get_pred_format()
            layer.cost_func = self.cost_func
            layer.optimiser_weight = copy(self.optimiser)
            layer.optimiser_bias = copy(self.optimiser)

        self.layers.append(layer)

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        epochs: int = 100,
        lam: float = 0,
        batches: int = 1,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ) -> dict:
        """
        Description:
        ------------
            Fits the CNN to input X for a given amount of epochs. Performs feedforward and backpropagation passes,
            can utilize batches, regulariziation and validation if desired.

        Parameters:
        ------------
            X (numpy array) with input data in format [images, input channels,
            image height, image_width]
            t (numpy array) target labels for input data
            epochs (int) amount of epochs
            lam (float) regulariziation term lambda
            batches (int) amount of batches input data splits into
            X_val (numpy array) validation data
            t_val (numpy array) target labels for validation data

        Returns:
        ------------
            scores (dict) a dictionary with "train_error", "train_acc", "val_error", val_acc" keys
            that contain numpy arrays with float values of all accuracies/errors over all epochs.
            Can be used to create plots. Also used to update the progress bar during training
        """

        # setup
        if self.seed is not None:
            np.random.seed(self.seed)

        # initialize weights
        self._initialize_weights(X)

        # create arrays for score metrics
        scores = self._initialize_scores(epochs)

        assert batches <= t.shape[0]
        batch_size = X.shape[0] // batches

        try:
            for epoch in range(epochs):
                for batch in range(batches):
                    # minibatch gradient descent
                    # If the for loop has reached the last batch, take all thats left
                    if batch == batches - 1:
                        X_batch = X[batch * batch_size :, :, :, :]
                        t_batch = t[batch * batch_size :, :]
                    else:
                        X_batch = X[
                            batch * batch_size : (batch + 1) * batch_size, :, :, :
                        ]
                        t_batch = t[batch * batch_size : (batch + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(t_batch, lam)

                # reset optimisers for each epoch (some optimisers pass in this call)
                for layer in self.layers:
                    if isinstance(layer, FullyConnectedLayer):
                        layer._reset_optimiser()

                # computing performance metrics
                scores = self._compute_scores(scores, epoch, X, t, X_val, t_val)

                # printing progress bar
                print_length = self._progress_bar(
                    epoch,
                    epochs,
                    scores,
                )
        # allows for stopping training at any point and seeing the result
        except KeyboardInterrupt:
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            epochs,
            epochs,
            scores,
        )
        sys.stdout.write("")

        return scores

    def _feedforward(self, X_batch) -> np.ndarray:
        """
        Description:
        ------------
            Performs the feedforward pass for all layers in the CNN. Called from fit()
        """
        a = X_batch
        for layer in self.layers:
            a = layer._feedforward(a)

        return a

    def _backpropagate(self, t_batch, lam) -> None:
        """
        Description:
        ------------
            Performs backpropagation for all layers in the CNN. Called from fit()
        """
        assert len(self.layers) >= 2
        reversed_layers = self.layers[::-1]

        # for every layer, backwards
        for i in range(len(reversed_layers) - 1):
            layer = reversed_layers[i]
            prev_layer = reversed_layers[i + 1]

            # OutputLayer
            if isinstance(layer, OutputLayer):
                prev_a = prev_layer.get_prev_a()
                weights_next, delta_next = layer._backpropagate(t_batch, prev_a, lam)

            # FullyConnectedLayer
            elif isinstance(layer, FullyConnectedLayer):
                assert (
                    delta_next is not None
                ), "No OutputLayer to follow FullyConnectedLayer"
                assert (
                    weights_next is not None
                ), "No OutputLayer to follow FullyConnectedLayer"
                prev_a = prev_layer.get_prev_a()
                weights_next, delta_next = layer._backpropagate(
                    weights_next, delta_next, prev_a, lam
                )

            # FlattenLayer
            elif isinstance(layer, FlattenLayer):
                assert (
                    delta_next is not None
                ), "No FullyConnectedLayer to follow FlattenLayer"
                assert (
                    weights_next is not None
                ), "No FullyConnectedLayer to follow FlattenLayer"
                delta_next = layer._backpropagate(weights_next, delta_next)

            # Conv2DLayer and Conv2DLayerOPT
            elif isinstance(layer, Conv2DLayer):
                assert (
                    delta_next is not None
                ), "No FlattenLayer to follow Convolution2DLayer"
                delta_next = layer._backpropagate(delta_next)
            elif isinstance(layer, Conv2DLayerOPT):
                assert (
                    delta_next is not None
                ), "No FlattenLayer to follow Convolution2DLayer"
                delta_next = layer._backpropagate(delta_next)

            # MaxPooling2DLayer
            elif isinstance(layer, MaxPooling2DLayer):
                assert delta_next is not None, "No Layer to follow MaxPooling2DLayer"
                delta_next = layer._backpropagate(delta_next)
            elif isinstance(layer, AveragePooling2DLayer):
                assert delta_next is not None, "No Layer to follow MaxPooling2DLayer"
                delta_next = layer._backpropagate(delta_next)

            # Catch error
            else:
                raise NotImplementedError

    def _compute_scores(
        self,
        scores: dict,
        epoch: int,
        X: np.ndarray,
        t: np.ndarray,
        X_val: np.ndarray,
        t_val: np.ndarray,
    ) -> dict:
        """
        Description:
        ------------
            Computes scores such as training error, training accuracy, validation error
            and validation accuracy for the CNN depending on if a validation set is used
            and if the CNN performs classification or regression

        Returns:
        ------------
            scores (dict) a dictionary with "train_error", "train_acc", "val_error", val_acc" keys
            that contain numpy arrays with float values of all accuracies/errors over all epochs.
            Can be used to create plots. Also used to update the progress bar during training
        """

        pred_train = self.predict(X)
        cost_function_train = self.cost_func(t)
        train_error = cost_function_train(pred_train)
        scores["train_error"][epoch] = train_error

        if X_val is not None and t_val is not None:
            cost_function_val = self.cost_func(t_val)
            pred_val = self.predict(X_val)
            val_error = cost_function_val(pred_val)
            scores["val_error"][epoch] = val_error

        if self.pred_format != "Regression":
            train_acc = self._accuracy(pred_train, t)
            scores["train_acc"][epoch] = train_acc
            if X_val is not None and t_val is not None:
                val_acc = self._accuracy(pred_val, t_val)
                scores["val_acc"][epoch] = val_acc

        return scores

    def _initialize_scores(self, epochs) -> dict:
        """
        Description:
        ------------
            Initializes scores such as training error, training accuracy, validation error
            and validation accuracy for the CNN

        Returns:
        ------------
            A dictionary with "train_error", "train_acc", "val_error", val_acc" keys that
            will contain numpy arrays with float values of all accuracies/errors over all epochs
            when passed through the _compute_scores() function during fit()
        """
        scores = dict()

        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        scores["train_error"] = train_errors
        scores["val_error"] = val_errors
        scores["train_acc"] = train_accs
        scores["val_acc"] = val_accs

        return scores

    def _initialize_weights(self, X: np.ndarray) -> None:
        """
        Description:
        ------------
            Initializes weights for all layers in CNN

        Parameters:
        ------------
            I   X (np.ndarray) input of format [img, feature_maps, height, width]
        """
        prev_nodes = X
        for layer in self.layers:
            prev_nodes = layer._reset_weights(prev_nodes)

    def predict(self, X: np.ndarray, *, threshold=0.5) -> np.ndarray:
        """
        Description:
        ------------
            Predicts output of input X

        Parameters:
        ------------
            I   X (np.ndarray) input [img, feature_maps, height, width]
        """

        prediction = self._feedforward(X)

        if self.pred_format == "Binary":
            return np.where(prediction > threshold, 1, 0)
        elif self.pred_format == "Multi-class":
            class_prediction = np.zeros(prediction.shape)
            for i in range(prediction.shape[0]):
                class_prediction[i, np.argmax(prediction[i, :])] = 1
            return class_prediction
        else:
            return prediction

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): output of predict() fuction
            (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (What the network should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))

    def _progress_bar(self, epoch: int, epochs: int, scores: dict) -> int:
        """
        Description:
        ------------
            Displays progress of training
        """
        progression = epoch / epochs
        epoch -= 1
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key, score in scores.items():
            if np.isnan(score[epoch]) == False:
                value = self._fmt(score[epoch], N=4)
                line += f"| {key}: {value} "
        print(line, end="\r")
        return len(line)

    def _fmt(self, value: int, N=4) -> str:
        """
        Description:
        ------------
            Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= N - 1:
            return str(round(value))
            # or overflow
        return f"{value:.{N-n-1}f}"
    
class VGG:
    def __init__(self):
        self.optimizer = AdamOptimiser(learning_rate=1e-4, beta1=0.9, beta2=0.999)
        self.model = CNN(
            cost_func=CostCrossEntropy,
            optimiser=self.optimizer,
            seed=42,
        )
        self.model.add(Conv2DLayerOPT(
            input_channels=3,
            feature_maps=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            act_func=RELU,
            seed=42,
        ))
        self.model.add(Conv2DLayerOPT(
            input_channels=32,
            feature_maps=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            act_func=RELU,
            seed=42,
        ))
        self.model.add(MaxPooling2DLayer(
            kernel_size=(2, 2),
            stride=(2, 2),
            seed=42,
        ))
        self.model.add(FlattenLayer(
            act_func=RELU,
            seed=42,
        ))
        self.model.add(FullyConnectedLayer(128))
        self.model.add(OutputLayer(10, output_func=softmax))

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        epochs: int = 100,
        lam: float = 0,
        batches: int = 1,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ) -> dict:
        return self.model.fit(X, t, epochs, lam, batches, X_val, t_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)