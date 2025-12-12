# Copyright 2019 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from setuptools import setup, find_packages

setup(
    name='gym_scaling',
    version='0.0.1',
    description='Smart AutoScaler RL Agent for Cloud Resource Optimization',
    author='Mohammadarshya Salehibakhsh, Saumya Goyal',
    author_email='msalehib@uci.edu, saumyg3@uci.edu',
    packages=find_packages(),
    install_requires=[
        'gym>=0.14.0',
        'overrides>=1.9',
        'numpy>=1.17.1,<2.0.0',
        'stable-baselines3>=1.6.0',
        'torch>=1.12.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
