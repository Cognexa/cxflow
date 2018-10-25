from setuptools import setup

setup(name='cxflow',
      version='0.12.0',
      description='Smart machine learning trainer',
      long_description='Trainer of machine learning models that automatically manages the whole process of training,'
                       'saving and restoring models and much more',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='machine learning training framework',
      url='https://github.com/cognexa/cxflow',
      author='Cognexa Solutions s.r.o.',
      author_email='info@cognexa.com',
      license='MIT',
      packages=['cxflow',
                'cxflow.cli',
                'cxflow.datasets',
                'cxflow.hooks',
                'cxflow.models',
                'cxflow.utils',
                'cxflow.tests'
                ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=[line for line in open('requirements.txt', 'r').readlines() if not line.startswith('#')],
      entry_points={
          'console_scripts': [
              'cxflow=cxflow.entry_point:entry_point'
          ]
      })
