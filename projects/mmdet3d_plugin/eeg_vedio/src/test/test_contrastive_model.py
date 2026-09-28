import os
import sys
import yaml

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
sys.path.append(project_dir)

config_path = os.path.join(project_dir, 'cfgs', 'train_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

import src.models.contrastive_model as contrastive_model

model = contrastive_model.ContrastiveModel(config["model"])