#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


REQUIREMENTS_FILE = "requirements.txt"
README_FILE = "README.md"

# Requirements
with open(REQUIREMENTS_FILE) as handle:
    requirements = handle.read().strip().split("\n")

with open(README_FILE) as readme_file:
    readme = readme_file.read()

setup(
    name="pathml",
    package="pathml",
    packages=find_packages(),
    use_scm_version={
        "write_to": "pathml/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
    author="Adam G. Berman, William R. Orchard, Marcel Gehrung, Florian Markowetz",
    author_email="florian.markowetz@cruk.cam.ac.uk",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="PathML: a Python library for deep learning on whole-slide images",
    long_description=readme,
    license="GNU General Public License v3",
    include_package_data=True,
    keywords=["pathology"],
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=requirements,
    url="https://github.com/markowetzlab/pathml",
    project_urls={
        "Bug Tracker": "https://github.com/markowetzlab/pathml/issues",
        "Documentation": "https://github.com/markowetzlab/pathml-tutorial",
        "Source Code": "https://github.com/markowetzlab/pathml",
    },
    zip_safe=False,
)
