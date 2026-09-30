# Methodology

## Dataset

CIFAR-100 is used for 100-class image classification. The original training set is split into 90% training and 10% validation using a fixed seed of 42. The official CIFAR-100 test set is kept separate.

## Models

- ResNet-50 with the first convolution adapted for 32×32 images and a 100-class output layer.
- MobileNetV2 with its first convolution stride adapted for CIFAR-100 and a 100-class output layer.

Both architectures use ImageNet initialization in the implementation.

## Compression methods

### Unstructured pruning

Global L1 unstructured pruning is applied to convolutional and linear weights. Pruning masks are subsequently removed from the module, leaving zero-valued weights in the dense tensor.

This distinction matters: sparsity does not imply smaller dense checkpoints or faster dense kernels. Structured pruning or sparse-aware kernels would be required to translate sparsity into hardware-level efficiency gains.

### Post-training INT8 quantization

A `QuantStub`/`DeQuantStub` wrapper is used with PyTorch eager-mode static quantization and the `fbgemm` backend. Twenty calibration batches are used before conversion. Quantized evaluation is performed on CPU.

## Energy measurement

CodeCarbon's `EmissionsTracker` records estimated energy consumption during the measured section. The repository reports energy as an estimate rather than a direct wattmeter measurement.

## Reproducibility notes

The implementation fixes Python, NumPy, and PyTorch seeds. The historical results remain single-run observations; rerunning the code can produce different values because software versions, pretrained weights, drivers, and hardware may differ.
