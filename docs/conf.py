# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
import os
import os.path

if not os.path.isdir('_static'):
    os.makedirs('_static')

# Generate Doxygen docs.
subprocess.run(['doxygen', 'Doxyfile.in'])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Aluminum'
copyright = '2018, Lawrence Livermore National Security'
author = 'Lawrence Livermore National Laboratory'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['breathe']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

rst_prolog = """
.. |AlLogo| image:: ../al.svg
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = 'favicon.ico'

# Breathe configuration

breathe_projects = {'Aluminum': '_doxyout/xml/'}
breathe_default_project = 'Aluminum'
