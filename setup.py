from setuptools import find_packages, setup

setup(
    name="cogsci17_semflu",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'nengo',
        'numpy',
        'matplotlib',
        'pytry'
    ]
)
