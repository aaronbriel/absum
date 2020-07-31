import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'absum'
AUTHOR = 'Aaron Briel'
AUTHOR_EMAIL = 'aaronbriel@gmail.com'
URL = 'https://github.com/aaronbriel/absum'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Abstract Summarization for Data Augmentation'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'numpy>=1.19.1',
    'pandas>=1.1.0',
    'torch>=1.6.0',
    'transformers>=3.0.2'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )