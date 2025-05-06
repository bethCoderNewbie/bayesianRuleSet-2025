from setuptools import setup, find_packages

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='bayesian-rule-set',
    version='0.1.0',
    author='Beth Nguyen',
    author_email='beth88_career@gmail.com',
    description='Implementation of Bayesian Rule Set algorithm for interpretable classification',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/yourusername/bayesian-rule-set',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
    ],
)