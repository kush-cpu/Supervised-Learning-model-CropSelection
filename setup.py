from setuptools import setup, find_packages

setup(
    name='crop-prediction-ml',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=1.3.2',
        'numpy>=1.24.3',
        'pandas>=2.0.1',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'pyyaml>=6.0'
    ],
    author='Kushagra',
    author_email='kushagranigam550@gmail.com',
    description='Machine Learning Project for Crop Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)