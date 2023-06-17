# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code author="Hans Meine"

import numpy as np

import torch
import torch.nn as nn




#from dnn.streams.gin_ipa_debugging_utils import view_tensor


class GinIpaAugmentation():
    """
    Implements Global Intensity Non-linear augmentation (GIN) 
    and Interventional Pseudo-correlation Augmentation (IPA)
    Source: https://arxiv.org/pdf/2111.12525.pdf
    """

    def __init__(self,
                 dim: int = 3,
                 input_channels: int = 1,
                 gin_hidden_layers: int = 4,
                 gin_hidden_channels: int = 2,
                 ipa_application_probability: float = 0.6,
                 ipa_control_points_per_dimension: int = 5,
                 probability_of_application: float = 0.5,
                 p_kernel_size_1: float = 0.5,
                 data_key: str = 'data'
                 ):
        super().__init__()

        self._dim = dim
        self._gin_model = self._build_gin_model(dim, input_channels, gin_hidden_layers, gin_hidden_channels, p_kernel_size_1)
        self._probability_of_application = probability_of_application
        self._ipa_application_probability = ipa_application_probability
        self._ipa_control_points_per_dimension = ipa_control_points_per_dimension
        self.data_key = data_key
            
    def _build_gin_model(self, dim: int, input_channels: int, layers: int, hidden_channels: int, p_kernel_size_1: float = 0.5):
        model = GinNet(dim, input_channels, layers, hidden_channels, p_kernel_size_1)

        return model

    """ def _input_streams_changed(self) -> None:
        pass

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['gin_model'] = self._gin_model # TODO: check if the model can simply be saved
        state['ipa_application_probability'] = self._ipa_application_probability
        state['ipa_control_points_per_dimension'] = self._ipa_control_points_per_dimension
        state['dim'] = self._dim
        state['probability_of_application'] = self._probability_of_application
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._gin_model = state['gin_model']
        self._ipa_application_probability = state['ipa_application_probability']
        self._ipa_control_points_per_dimension = state['ipa_control_points_per_dimension']
        self._dim = state['dim']
        self._probability_of_application = state['probability_of_application']"""

    def __call__(self, **data_dict):
        # Works only on the first stream (supposed to be the image stream) so far

        gin_input = torch.from_numpy(data_dict[self.data_key])

        augmented_images = []
        for image in gin_input:
            image = image.unsqueeze_(dim=0)
            if np.random.uniform() < self._probability_of_application:
                with torch.no_grad():
                    if np.random.uniform() < self._ipa_application_probability:
                        gin_output_1 = self._gin_model.forward(image)
                        gin_output_2 = self._gin_model.forward(image)
                        output = self._apply_ipa(gin_output_1, gin_output_2)
                    else:
                        output = self._gin_model.forward(image)
            else:
                output = image
            augmented_images.append(output)

            #view_tensor(output)

        output = np.array(torch.cat(augmented_images, dim=0))        
        data_dict[self.data_key] = output
        return data_dict
        #return tuple([output] + [input_batch[i] for i in range(1, len(input_batch))])

    def _apply_ipa(self, gin_output_1, gin_output_2):
        # This is only a simplyfied version of the interpolation used in the paper
        grid_shape = (gin_output_1.shape[0], 1) + (self._ipa_control_points_per_dimension,) * self._dim
        random_grid_points = torch.rand(grid_shape)

        sigma = 2
        kernel_size = int(2 * sigma + 1)
        gaussian_kernel_1d = torch.tensor([torch.exp(-(x - sigma) ** 2 / (2 * sigma ** 2)) for x in torch.arange(0, kernel_size)])
        gaussian_kernel = gaussian_kernel_1d[None, :] * gaussian_kernel_1d[:, None]
        if self._dim == 3:
            gaussian_kernel = gaussian_kernel[None, :] * gaussian_kernel_1d[:, None, None]
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(1)

        if self._dim==2:
            resampled_field = nn.functional.interpolate(random_grid_points, size=gin_output_1.shape[2:], mode='bilinear')
            resampled_field = nn.functional.conv2d(resampled_field, gaussian_kernel, padding=kernel_size//2)
        elif self._dim==3:
            resampled_field = nn.functional.interpolate(random_grid_points, size=gin_output_1.shape[2:], mode='trilinear')
            resampled_field = nn.functional.conv3d(resampled_field, gaussian_kernel, padding=kernel_size//2)
        else:
            raise Exception(f'Only support 2 or 3 spatial dimensions, but got dim={self._dim}')

        output = resampled_field * gin_output_1 + (1 - resampled_field) * gin_output_2

        return output

    #def data_identifier_addition(self):
    #    return ['gin_ipa']


class GinNet(nn.Module):
    def __init__(self, dim: int, input_channels: int, hidden_layers: int, hidden_channels: int, p_kernel_size_1: float = 0.5):
        super(GinNet, self).__init__()

        self._p_kernel_size_1 = p_kernel_size_1
        
        self.layers_kernel_size_3 = nn.ModuleList()
        self.layers_kernel_size_1 = nn.ModuleList()
        self.layers_activation_functions = nn.ModuleList()
        
        layer_dims = [input_channels] + [hidden_channels]*(hidden_layers-1) + [input_channels]

        for i in range(len(layer_dims) - 1):
            if dim == 2:
                self.layers_kernel_size_3.append(nn.Conv2d(in_channels=layer_dims[i], out_channels=layer_dims[i+1], kernel_size=3, padding=1))
                self.layers_kernel_size_1.append(nn.Conv2d(in_channels=layer_dims[i], out_channels=layer_dims[i+1], kernel_size=1, padding=0))
            elif dim == 3:
                self.layers_kernel_size_3.append(nn.Conv3d(in_channels=layer_dims[i], out_channels=layer_dims[i+1], kernel_size=3, padding=1))
                self.layers_kernel_size_1.append(nn.Conv3d(in_channels=layer_dims[i], out_channels=layer_dims[i+1], kernel_size=1, padding=0))
            self.layers_activation_functions.append(nn.LeakyReLU())

    def forward(self, input):
        for layer in self.layers_kernel_size_3:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.uniform_(layer.bias)
        for layer in self.layers_kernel_size_1:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.uniform_(layer.bias)

        x = input
        for i in range(len(self.layers_kernel_size_3)):
            if np.random.random() < self._p_kernel_size_1:
                x = self.layers_kernel_size_1[i](x)
            else:
                x = self.layers_kernel_size_3[i](x)
            x = self.layers_activation_functions[i](x)

        alpha = torch.rand(1)
        output = alpha * input + (1-alpha) * x
        output = output * (torch.norm(input, p='fro') / torch.norm(output, p='fro'))

        return output
    