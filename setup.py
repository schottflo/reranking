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
          'sentence-transformers==2.0.0',
          'scikit-learn==0.24.2',
          'matplotlib==3.4.2',
          'conllu==1.3.1',
          'spanningtrees git+https://github.com/rycolab/spanningtrees',
          'joblib==1.0.1',
          'seaborn==0.11.1',
          'scipy==1.6.3',
          'pandas=1.2.4'
          ],
      packages=find_packages(),
      )