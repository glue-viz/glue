import sys

import yaml
import requests
from jinja2 import Template

sys.path.insert(0, '..')
import glue

RECIPE = 'https://raw.githubusercontent.com/conda-forge/glueviz-feedstock/master/recipe/meta.yaml'

content = requests.get(RECIPE).text

recipe = yaml.load(Template(content).render())

recipe['package']['version'] = glue.__version__
recipe['source'] = {'path': '../../'}

with open('recipe/meta.yaml', 'w') as f:
    yaml.dump(recipe, stream=f, default_flow_style=False)
