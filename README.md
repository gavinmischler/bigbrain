# bigbrain
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

bigbrain is a toolbox for learning anything and everything from data.

# No machine learning knowledge needed!

With this toolbox, all your data science problems can be solved with a
single command. No longer will you need to know anything about machine
learning in order to use its power to solve all your problems.

  - Learn the best models from your data without any background knowledge!
  - Let the AI learn the best algorithm for your data, and then use what it learned on other data!

### Tech

bigbrain wraps [scikit-learn](https://github.com/scikit-learn/scikit-learn) to try many different models on your data and find the best performing algorithm for you.

### Installation

bigbrain can be installed using pypi, either from the [website](https://pypi.org/project/bigbrain), or from the command line:

```sh
$ pip install bigbrain
```

### Example Usage

With this package, you can easily create regression or classification models for supervised learning techniques.

```py
>>> from bigbrain import AI

####### Classification #######
>>> machine = AI(model_type='classification')
>>> machine.learn(X_train, y_train)
"My big brain has learned everything."
>>> predictions = machine.go(X_test)

####### Regression #######
>>> machine = AI(model_type='regression')
"My big brain has learned everything."
>>> predictions = machine.go(X_test)
```
