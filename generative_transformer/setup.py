from setuptools import setup,find_packages,Command
import os

__version__ = "0.0.1"

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./src/*.egg-info')

with open("README.md", "r") as fh:
    long_description = fh.read() or ""

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(name="genet",
      version=__version__,
      description='generative ai with transformer',
      author='yduan',
      packages=find_packages("src"),
      package_dir={"": "src"},
      install_requires=['torch>=1.0', 'python>=10.0'],
      zip_safe=False,
      cmdclass={
        'clean': CleanCommand,
    }
      )


