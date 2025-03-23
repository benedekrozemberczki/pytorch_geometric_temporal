import datetime
import sphinx_rtd_theme
import doctest
import inspect





extensions = [
    'autoapi.extension',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
]

source_suffix = '.rst'
master_doc = 'index'

autoapi_python_use_implicit_namespaces = False

author = 'Benedek Rozemberczki'
project = 'PyTorch Geometric Temporal'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

html_theme = 'sphinx_rtd_theme'
autoapi_add_toctree_entry = True

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
     'navigation_depth': 2,
}


html_logo = '_static/img/text_logo.jpg'
html_static_path = ['_static']

add_module_names = False
autoapi_generate_api_docs = False

# --- AutoAPI config ---
autoapi_type = 'python'
autoapi_dirs = ['../../torch_geometric_temporal/']