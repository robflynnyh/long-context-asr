from setuptools import setup, find_packages

setup(
    name='lcasr',
    version='1.0',    
    description='Code for training long-context asr models on spotify podcast corpus',
    url='https://github.com/robflynnyh/long-context-asr',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ],
)
