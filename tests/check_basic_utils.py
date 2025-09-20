from disentangler_utils import *


def main():
    """
    This test checks that the composition maps A and A_inv are inverses.

    To run, add the parent directory for testing
    export PYTHONPATH=(absolute path to parent):$PYTHONPATH 
    """
    np.random.seed(40)
    
    # A and A_inv are inverses
    l = 2
    r = 3
    b = 5
    c = 7

    X = np.random.randn(l*r, b*c)
    X = X/np.linalg.norm(X)
    
    AX = A(X, l, r, b, c)
    Xp = A_inv(AX, l, r, b, c)
    
    if np.linalg.norm(X - Xp) < 1e-14:
        print('A and A_inv are inverses')
    
    
if __name__ == '__main__':
    main()


