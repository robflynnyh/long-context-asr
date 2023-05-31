from setuptools import setup

setup(
    name='lcasr',
    version='0.0.0.1',    
    description='Code for training long-context asr models on spotify podcast corpus',
    url='https://github.com/robflynnyh/long-context-asr',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='MIT',
    packages=['lcasr'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
