import datetime
import sphinx_rtd_theme
import doctest

extensions = [
    'sphinx.ext.autodoc',
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
# autoapi_ignore = ["*"]

autoapi_ignore = ["*temporal.signal*", "*temporal.nn*", "*temporal.datasets*"]

author = 'Benedek Rozemberczki'
project = 'PyTorch Geometric Temporal'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

html_theme = 'sphinx_rtd_theme'
autoapi_add_toctree_entry = False

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
# html_context = {'css_files': ['_static/css/custom.css']}

add_module_names = False
# autoapi_generate_api_docs = False
# --- AutoAPI config ---
autoapi_type = 'python'
autoapi_dirs = ['../../torch_geometric_temporal']  # Adjust this to the relative path of your source code
autoapi_keep_files = True  # Optional: keeps intermediate .rst files
autoapi_options = [
    'members',
]

# Optional filtering (like the original skip list)
# You can use custom templates or post-process the .rst files to exclude members like __init__, etc.
