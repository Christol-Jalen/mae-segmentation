# SparK Pre-training Framework

## Introduction

This repository contains the SparK Pre-training Framework, which is designed for distributed pre-training of sparse encoder-decoder neural network architectures. The framework is particularly optimized for image-based models and is capable of handling large-scale datasets efficiently.

## Environment Setup

### Requirements

To run the SparK Pre-training Framework, you need the following software and libraries:

- Python 3.8 or higher
- PyTorch 1.8 or higher
- Other Python libraries as specified in `requirements.txt`

### Installation

1. **Clone the repository:**

```sh
git clone https://github.com/Christol-Jalen/Applied-Deep-Learning-CW3.git
cd [Repository Directory]
```

2. **Set up a Python virtual environment (recommended):**

```sh
python -m venv spark-env
source spark-env/bin/activate  # On Windows use `spark-env\Scripts\activate`
```

3. **Install the required packages:**

```sh
pip install -r requirements.txt
```

### Configuration

Before starting the pre-training, you need to configure the following:

- `args.data_path`: Path to the training data.
- `args.input_size`: Size of the input images.
- `args.dataloader_workers`: Number of worker processes for data loading.
- `args.glb_batch_size`: Global batch size for distributed training.
- `args.batch_size_per_gpu`: Batch size per GPU.
- `args.model`: Model architecture to use.
- `args.sbn`: Whether to use SyncBatchNorm.
- `args.dp`: Drop path rate for the encoder.
- `args.mask`: Mask ratio for the model.

These configurations are typically set in the `arg_util.Args` class or passed via the command line.

### Running the Pre-training Script

To run the pre-training, execute the following command:

```sh
python main.py
```

Make sure the distributed training environment is correctly set up if you plan to use multiple GPUs or nodes.

## Additional Notes

- The training process leverages DistributedDataParallel (DDP) for efficient multi-GPU training.
- The script is set up to automatically resume training from the last checkpoint if available.
- Real-time logging can be monitored using TensorBoard if configured.

## Contributing

Contributions to improve the framework are welcome. Please adhere to the existing coding style and add comments where necessary.

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
