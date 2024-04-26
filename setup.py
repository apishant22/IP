from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Streaming video data via networks'
LONG_DESCRIPTION = 'A package that allows to build simple streams of video, audio and camera data.'

# Setting up
setup(
    name="framework",
    version=VERSION,
    author="Apishan",
    author_email="<at16g21@soton.ac.uk>",
    description=DESCRIPTION,
    url='https://github.com/apishant22/IP.git',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'pandas',
        'Pillow',
        'tqdm',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'captum',
        'transformers'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)