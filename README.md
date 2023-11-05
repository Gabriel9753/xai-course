# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

[[`Paper`](https://arxiv.org/abs/1610.02391)]

## Algorithm Explained
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used in the field of computer vision and deep learning to understand the reasoning behind the decisions made by convolutional neural networks (CNNs) in visual tasks. It does so by producing a coarse localization map highlighting the important regions in the image for predicting the concept of interest (see figure below).
![Algorithm](assets/graphics_1.png)
The algorithm is based on the idea that the importance of a feature map can be computed by the gradient of the score for a given concept (e.g. class) with respect to the feature map. The gradients are global average pooled to obtain the neuron importance weights, which are then used to linearly combine the feature maps to obtain the coarse localization map.

The algorithm can be divided into the following steps:
1. Feature Map Generation: First, the forward pass is performed on the CNN, and the activation maps of the final convolutional layer are obtained.

2. Gradient Calculation: Next, the gradients of the target class (the class for which the network's decision is being analyzed) with respect to the final convolutional layer are computed.

3. Global Average Pooling (GAP): The gradients are then globally averaged to obtain the importance weights for each feature map.

4. Activation Map Localization: These importance weights are multiplied by their corresponding feature maps, generating the GradCAM heatmap. This heatmap highlights the important regions in the input image that contributed the most to the final classification decision.

## Ressources
[[`Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization`](https://arxiv.org/abs/1610.02391)]

[[`Pytorch Implementation`](https://github.com/jacobgil/pytorch-grad-cam)]