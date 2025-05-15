# tests/conftest.py
import sys, os

# __file__: /.../hw2/tests/conftest.py
# 我們要把 '/.../hw2' 加到 sys.path，這樣就可以 import src.loader 了
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
