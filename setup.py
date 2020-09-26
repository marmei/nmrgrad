from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='nmrgrad',
    version='0.0.2',
    description='Tools for gradient coil calculation',
    author='Markus Meissner',
    # packages=['biot-savart'],
    install_requires=['numpy'], #external packages as dependencies
    # url='https://pip.pypa.io/',
    # packages=find_namespace_packages(include=['nmrgrad.*']),
    packages=find_packages(),
    keywords='micro gradient mri nmr design klayout',
    package_dir={'nmrgrad': 'nmrgrad'},
    include_package_data=True,
)
