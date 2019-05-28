from setuptools import setup, find_packages

setup(
    name='delira-enas',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/justusschock/denas',
    license='',
    author='Justus Schock',
    author_email='',
    description='',
    install_requires=["delira">=0.3.3, "torch>=1.0.0"]
)
