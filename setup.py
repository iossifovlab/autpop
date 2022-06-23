"""Setup for autism population modules package"""

import setuptools


setuptools.setup(
    name="autpop",
    version="0.7",
    author="Ivan Iossifov",
    author_email="iossifov@cshl.edu",
    description="Autism population model",
    url="https://github.com/iossifov/graphs",
    packages=['autpop'],
    include_package_data=False,
    entry_points="""
    [console_scripts]
    autpop=autpop.population_threshold_model:cli
    """,
    classifiers=[
        "Development Status :: Beta",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
