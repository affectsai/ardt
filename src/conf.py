# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
from pathlib import Path
os.environ['ARDT_CONFIG_PATH']="/mnt/affectsai/arrc-ardt-training/datasets/ardt_config.yaml"
sys.path.insert(0, os.path.abspath('ardt'))

project = 'Affects Research Dataset Toolkit (ARDT)'
copyright = '2025, Affects AI, LLC'
author = 'Affects AI, LLC'
release = '0.3.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# extensions = ['autoapi.extension']
# # Document Python Code
# autoapi_type = 'python'
# autoapi_dirs = ['./ardt']


templates_path = ['_templates']
exclude_patterns = ['__about__.py']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
