# About

This directory contains the code and data necessary to train a simple multi-layer perceptron (MLP) on the provided hearing loss dataset.

There are two Python files provided, `train_3dscan.py` and `train_imageslice.py`. As the names imply, one file is a training script for training the MLP with the entire flattened 3D scan passed as input, whereas the other is a training script where horizontal 2D image slices are passed as input instead.



# Dependencies

To install all necessary dependencies, run `pip install -r requirements.txt`



# Usage

Weights & Biases Login

These scripts rely on W&B for logging. As such, you need to have already created a W&B account. Then, in your terminal, run the command `wandb login` to enter your API key.



### <u>Individual Run</u>

You can perform an individual training run with the following commands:

`python3 train_3dscan.py`

`python3 train_imageslice.py`

Both files support the following arguments to adjust hyperparameters: `learning_rate`, `batch_size`, `min_lr_divisor`, `dropout_p`, `epochs`, `validation_split`, `log_iter`, and `zoom_factor`.


### <u>**Sweep**</u>

To automatically iterate over several possible hyperparameters to determine the best choice for your model, you can make use of W&B Sweeps. The configuration files for these sweeps can be found at `sweep_config_3dscan.yaml` and `sweep_config_imageslice.yaml`. These contain the different hyperparameters that are iterated over through training, and can be used to replicate our MLP results.

You can run a sweep with the following command:

`wandb sweep --project <project-name> <path-to-yaml-config-file>`


Then, run the `wandb agent` command that is provided in the output of the previous command.
