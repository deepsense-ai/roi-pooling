#!/usr/bin/env python

from distutils.core import setup
from distutils.command.install import install as DistutilsInstall

import subprocess

class MyInstall(DistutilsInstall):
    def run(self):
        subprocess.call(['make', '-C', 'roi_pooling', 'build'])
        DistutilsInstall.run(self)

setup(name='roi-pooling',
            version='1.0',
            description='ROI pooling as a custom TensorFlow operation',
            author='deepsense.io',
            packages=['roi_pooling'],
            package_data={'roi_pooling': ['roi_pooling.so']},
            cmdclass={'install': MyInstall}
)


