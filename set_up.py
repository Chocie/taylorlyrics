from setuptools import setup, find_packages

setup(
    name='taylorlyrics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'nltk',
        'keras',
        'tensorflow',
        'matplotlib',
        'textblob'
    ],
    author='Ziwei Jin',
    author_email='zoej7729@gmail.com',
    description='A package for processing Taylor Swift lyrics and training RNN models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Chocie/taylorlyrics',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
