"""

This module implements a class CustomEval that is a callable and inherit
from float. It returns itself (a float and an array) when called, so that it
can be used as a scoring function, but the float it represents is meaningless.
Instead, it stores the eval evaluation in a field named `eval`, so that it
can be accessed later.

This is needed because sklearn for some reason wants that the scoring function
returns a number

"""

from sklearn.metrics import confusion_matrix
import numpy as np

asarray_back = np.asarray


def new_asarray(*args, **kwargs):
    if type(args[0]) in [CustomEval, CustomEvalFloat]:
        return args[0]
    if type(args[0]) is list and len(args[0]) > 0:
        if type(args[0][0]) in [CustomEval, CustomEvalFloat]:
            return args[0]
    return asarray_back(*args, **kwargs)


np.asarray = new_asarray

array_back = np.array


def new_array(*args, **kwargs):
    if type(args[0]) in [CustomEval, CustomEvalFloat]:
        return args[0]
    if type(args[0]) is list:
        if type(args[0]) in [CustomEval, CustomEvalFloat]:
            np.stack(args[0])
    return array_back(*args, **kwargs)


np.array = new_array


class CustomEval(np.ndarray):

    __name__ = 'custom_eval_hack'

    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def __call__(self, *args, **kwargs):
        true = args[0]
        pred = args[1]
        self.eval = {}
        self.eval['matrix'] = confusion_matrix(*args, **kwargs, labels=self.labels)
        if hasattr(true, 'index'):
            self.eval['wrong'] = true.index[true != pred]
            self.eval['indices'] = true.index
        self.eval['predicted'] = pred
        return self

    def item(self, *args, **kwargs):
        obj = CustomEvalFloat()
        obj.eval = self.eval
        return obj

    def __array_finalize__(self, obj):
        self.eval = getattr(obj, 'eval', None)
        return self

    def __new__(cls, *args, **kwargs):
        if args:
            obj = cls(labels=args[0])
        else:
            obj = np.ndarray.__new__(cls, 1)
        return obj

    def __reduce__(self):
        return CustomEval, (self.labels, ), {
            'eval': self.eval,
        }

    def __setstate__(self, state):
        self.eval = state['eval']


class CustomEvalFloat(float):

    def __init__(self):
        super().__init__()

    def __sum__(self, x):
        return self.eval['matrix'] + x.eval['matrix']

    def __repr__(self):
        return "CustomEvalDummyFloat"
