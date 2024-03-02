Implementation of the [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) and the [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) on it.

# Dataset
The model is trained to generate images similar to the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset of celebrity faces. The dataset contains 202599 images of cropped and aligned faces.

# Models
## Generator
![image](https://github.com/hanialnahas/dcgan-pytorch/assets/41491376/aee4a7eb-6707-41b4-bd8e-635e09136dfe)
The generator network consists of 5 conv layers with a latent vector input and an output size of 64x64

## Discriminator
![image](https://github.com/hanialnahas/dcgan-pytorch/assets/41491376/06e735e0-b6ca-4ccf-b234-79f5bd4b12c3)
The discriminator network consists of 5 conv layers with a 64x64 image input and outputs a sigmoid probability of the image being fake or real.

# Training
The model is trained for 5 epochs with the hyperparameters defined in the [paper](https://arxiv.org/pdf/1511.06434.pdf)

# Result
![image](https://raw.githubusercontent.com/hanialnahas/dcgan-pytorch/master/output.png)
