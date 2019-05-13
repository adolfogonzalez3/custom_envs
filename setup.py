from setuptools import setup, find_packages
import sys


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='custom_envs',
      packages=[package for package in find_packages()
                if package.startswith('custom_envs')],
      install_requires=[
          'numpy',
      ],
      extras_require={
        
      },
      description='A set of environments used for the ML Lab.',
      author='Adolfo Gonzalez III',
      url='',
      author_email='',
      keywords="",
      license="",
      )

