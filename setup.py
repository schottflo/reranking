from setuptools import setup, find_packages

setup(name='parsing',
      version='1.0',
      description='Methods for Reranking Parse Tree',
      author=['Florian Schottmann'],
      url='https://github.com/schottflo/bayesian-nlp',
      install_requires=[
          'numpy==1.20.3',
          'nltk==3.6.2',
          'GPy==1.10.0',
          'autograd==1.3',
          'stanza==1.2.2',
          'sentence-transformers==1.1.0',
          'scikit-learn==0.24.2',
          'matplotlib==3.4.2'
          ],
      packages=find_packages(),
      )