from setuptools import setup,find_packages

setup(name="basic_transformer",
      version='0.1',
      description='pytorch implementation of transformer',
      author='yduan',
      packages=find_packages(),
      install_requires=['torch>=1.0'],
      zip_safe=False
      )
