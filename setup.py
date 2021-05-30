from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="my-optimizer",
    version="0.1",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["my_optimizer"],
    install_requires=[
        "tensorflow~=2.5",
    ],
    entry_points="""
        [tensorflow.optimizers]
        my_optimizer = my_optimizer:MyOptimizer
    """ 
)
