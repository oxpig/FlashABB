from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name='flash-abb',
    version='0.0.4',
    license='BSD 3-clause license',
    description='FlashABB: modelling antibody structures at the speed of language',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Isaac Ellmen",
    maintainer='Isaac Ellmen',
    maintainer_email='isaac.ellmen@stats.ox.ac.uk',
    url="https://github.com/oxpig/FlashABB",
    include_package_data=True,
    packages=find_packages(include=('flash_abb', 'flash_abb.*')),
    package_data={'flash_abb': ['weights/*']},
    install_requires=[
        'torch>2',
        'requests',
        'einops',
        'rotary-embedding-torch',
        'ml_collections',
        'numpy',
        'dm-tree',
        'pyyaml',
        'scipy',
    ],
)
