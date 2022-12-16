from setuptools import setup

setup(
    name='dappy',
    version='0.1_dev',    
    description='3D Behavioral Analysis of Action Skeletons',
    url='https://github.com/joshuahwu/dappy',
    author='Joshua Wu',
    author_email='joshua.wu@duke.edu',
    license='BSD 2-clause',
    packages=[''],
    install_requires=['scipy',
                      'pandas',
                      'matplotlib',
                      'numpy',
                      'sklearn',                    
                      'pyyaml',
                      'h5py',
                      'hdf5storage',
                      'tqdm',
                      'fbpca'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)