from setuptools import setup

setup(
    name='lcasr',
    version='0.0.9',    
    description='Code for training long-context asr models on spotify podcast corpus',
    url='https://github.com/robflynnyh/long-context-asr',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='Apache 2.0',
    packages=['lcasr'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ],
)
