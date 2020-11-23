# Dataset-Distilllation
Implementation of the technique described in the paper "Dataset Distillation (2018)" to generate distilled image for a model with a fixed initialization.

## Installation
For the code to work you'll have to install
- ```matplotlib``` - used to save and visualized the distilled images 
- ```higher``` - pytorch package for dealing with higher-order gradients
- ```dill``` - to pickle and unpickle the lambdas contained in the configuration file
- ```continual-flame``` - contains continual learning utilities and datasets (used to manage the distilled images)
- ```pytorch```

## Structure

### Distill
The ```distill.py``` file contains all the logic necessary to generate distilled examples. The code is based on the algorithm described in the paper "Datase Distillation", or at least on my understanding of it.

Setting an output path different from None in the config file will lead to ```distill.py``` writing the distilled dataset in output.

### Test
Running the file ```test.py``` you can train a model on the distilled dataset produced by ```distill.py```. The model will be trained for the specified number of epochs and tested on distilled dataset's test set. The results are printed to stdout.

### Config
The configurations file contain all the information necessary to run a given experiment (e.g. hyperparameters, log level, dataset, etc.). To execute a configuration just run ```python <config-name>.py```. Depending on the configuration and to the hardware you are using, the complition time may vary from few minutes to several hours.

## Results
The algorithm, applied to the MNIST dataset, can compress it up to just one image per class. The resulting images are sufficient to get the target model to 97% of accuracy.
Experiment on CIFAR10 and CIFAR100 are yet to be done. I'll upload the corresponding configuration once I run them and get good results.

## Contribute
The repository is open to contributions. E.g. adding new configurations, eliminating the dependency from ```continual-flame```,...
