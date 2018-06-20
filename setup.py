from setuptools import setup, find_packages
import os

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 0, '')
__version__ = '%d.%d.%d%s' % version_info


setup(name='montage',
      version=__version__,
      description='Image Montage maker',
      url='http://github.com/pbmanis/cnmodel',
      author='Paul B. Manis',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['montage*']),
      zip_safe=False)
      