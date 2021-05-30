# My Custom Tensorflow Optimizer

This is a template mockup for for a custom tensorflow optimizer see [Plugin-able
Optimizers #49895](https://github.com/tensorflow/tensorflow/issues/49895).
Replace this with a short description of your Optimizer.

## Template Usage

1. Replace all occurances of `my-optimizer` with your `pip` package name
2. Replace all occurances of `MyOptimizer` with the class name of your optimizer
   and fix the `**hyperparemeters` in this README.
3. Replace all occurances of `my_optimizer` with the name of your implementation
   file (or package) and rename the implementation template.
4. Write your Optimizer
5. Delete this Section from the `README` and modify the remaining sections

## Installation

```bash
pip install my-optimizer
```

## Usage

```python
import tensorflow as tf
from tensorflow.optimizers import MyOptimizer

inputs = tf.keras.layers.Input(shape=(3,))
outputs = tf.keras.layers.Dense(2)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
   optimizer=MyOptimizer(**hyperparameters), 
   loss="mse", 
   metrics=["mae"]
)
```

## Details

Mathematical Details of the Optimizer.


## How to Cite

Bibtex / DOI / etc.
