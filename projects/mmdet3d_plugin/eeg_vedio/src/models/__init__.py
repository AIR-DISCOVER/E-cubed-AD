import sys
import os
models_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(models_dir, '..'))
project_dir = os.path.abspath(os.path.join(src_dir, '..'))

# sys.path.append(src_dir)
sys.path.append(project_dir)