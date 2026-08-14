# Image Colorization using VAEs and GANs

This ECE1508 course project compares deep-learning approaches for converting grayscale images into color. Image colorization is an ambiguous problem: a grayscale object can often have several plausible colors, so a model must produce convincing chrominance while preserving the structure of the input image.

We compare three approaches:

- a deterministic baseline;
- a conditional variational autoencoder (CVAE); and
- a conditional generative adversarial network (CGAN).


## Data representation

The experiments use CIFAR-10 with a fixed split of 45,000 training, 5,000 validation, and 10,000 test images. RGB images are converted to the CIELAB color space. Each model receives the luminance channel, `L`, as its grayscale input and predicts the two chrominance channels, `ab`:

```text
L [1 x 32 x 32] -> model -> ab [2 x 32 x 32]
```

The predicted `ab` channels are combined with the original `L` channel and converted back to RGB for visualization and evaluation. All models use the same preprocessed HDF5 files, splits, and normalization so that their results are directly comparable. The shared preprocessing definition is provided in [`preprocess_image_colorization_data.py`](preprocess_image_colorization_data.py); model implementations consume the prepared data rather than preprocessing it again.

## Evaluation

The models are compared using:

- normalized `ab`-space mean squared error (MSE) and mean absolute error (MAE);
- RGB peak signal-to-noise ratio (PSNR);
- RGB structural similarity index (SSIM); and
- qualitative comparisons of grayscale inputs, predictions, and ground-truth images.

Together, these measurements assess reconstruction error, structural consistency, and the visual plausibility of the predicted colors.

## Repository structure

```text
CNN/                                  Deterministic baseline(autoencoder)
GAN/                                  Conditional GAN implementation
preprocess_image_colorization_data.py Shared preprocessing definition
```

Detailed setup, training, tuning, visualization, and evaluation instructions for the deterministic baseline are available in [`CNN/README.md`](CNN/README.md).

