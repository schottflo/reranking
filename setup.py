from setuptools import setup, find_packages

setup(name='parsing',
      version='1.0',
      description='Methods for Reranking Parse Tree',
      author=['Florian Schottmann'],
      url='https://github.com/schottflo/bayesian-nlp',
      install_requires=[
          'numpy',
          'nltk',
          'pyStatParser',
          'cfgen',
          'autograd',
          'stanza'
      ],
      packages=find_packages(),
      )