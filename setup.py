"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='bittensor',  # Required
    version='0.0.1',  # Required
    description='Decentralized Machine Intelligence Deamon',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/unconst/bittensor',  # Optional
    author='Jacob R. Steeves',  # Optional
    author_email='jrsteeves@live.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(
        exclude=['data', 'contract', 'assets', 'scripts', 'docs']),  # Required
    python_requires='>=3.5',
    install_requires=[
        'argparse', 'grpcio', 'grpcio-tools', 'libeospy', 'loguru',
        'matplotlib', 'miniupnpc', 'networkx', 'numpy', 'pebble',
        'pickle-mixin', 'pycrypto', 'sklearn', 'tensorflow==1.15.0',
        'tensorflow_hub==0.4.0', 'timeloop', 'zipfile36'
    ],  # Optional
)
