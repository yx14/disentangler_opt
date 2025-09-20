To run tests, the parent directory containing the disentangler_utils.py file must be added to the python path:

export PYTHONPATH=(absolute path to parent directory):$PYTHONPATH 

Usage:

```
python check_basic_utils.py
python check_gradient_and_hessian.py 
```

Note: the gradient and hessian tests (from pymanopt) include plotting and require matplotlib
