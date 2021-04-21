from setuptools import setup, find_packages

setup(name='spankernel',
      version='1.0',
      description='Algorithms for Bayesian NLP',
      author=['Florian Schottmann'],
      url='https://github.com/schottflo/bayesian-nlp',
      install_requires=[
          'numpy',
          'nltk',
          'pyStatParser',
          'cfgen',
      ],
      packages=find_packages(),
      )