import datetime
import sphinx_rtd_theme
import doctest

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'autoapi.extension',
]

# AutoAPI configuration
autoapi_type = 'python3'
autoapi_dirs = ['../../torch_geometric_temporal']  # Update path if needed

source_suffix = '.rst'
master_doc = 'index'

author = 'Benedek Rozemberczki'
project = 'PyTorch Geometric Temporal'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

autoapi_generate_api_docs = False
doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_logo = '_static/img/text_logo.jpg'
html_static_path = ['_static']
html_context = {'css_files': ['_static/css/custom.css']}

add_module_names = False

def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)
