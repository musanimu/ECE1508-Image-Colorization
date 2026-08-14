# ECE1508 Image Colorization

Course project comparing three approaches to CIFAR-10 image colorization:

- deterministic CNN baseline
- conditional GAN
- conditional VAE

All models use the same preprocessed CIELAB data. The luminance channel is provided as input, and each model predicts the missing chrominance channels.

The deterministic baseline implementation and reproduction instructions are available in [CNN/README.md](CNN/README.md).
