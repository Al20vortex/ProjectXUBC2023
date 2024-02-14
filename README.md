### Self Expanding CNN (ProjectX UBC submission)
***
This is the official implementation of the [Self Expanding Convolutional Neural Network](https://arxiv.org/abs/2401.05686) paper, which is our submission to the ProjectX machine learning research competition organized by `UofT AI`. 
***
**Project Structure**
```python
├─models
  ├── __init__.py
  ├── convBlock.py
  ├── dynamicCNN.py
  ├── identityConv.py
  ├── perceptron.py
├─utils
  ├── __init__.py
  ├── train.py
  ├── utils.py
├─ dataloaders.py
├─ main.py
├─ README.md
```

**To Recreate the results**:
- clone the current repository
- Install the following dependencies in a virtual environment:
  - pytorch
  - wandb 
  - tqdm
- Run the `main.py` file from the root directory

We benchmarked on the [CIFAR 10](https://en.wikipedia.org/wiki/CIFAR-10) dataset. You can easily create your own custom dataset and pass the dataloaders into the `train_model` function. 
***
HAPPY CODING!! 