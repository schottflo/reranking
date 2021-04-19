from setuptools import setup, find_packages

setup(name='bayesian-nlp',
      version='1.0',
      description='Algorithms for Bayesian NLP',
      author=['Florian Schottmann'],
      url='https://github.com/schottflo/bayesian-nlp',
      install_requires=[
          'numpy',
          'nltk',
          'cfgen'
      ],
      packages=find_packages(),
      )