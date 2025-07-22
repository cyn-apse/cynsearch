from setuptools import setup, find_packages

setup(
    name="cynsearch",
    version="0.1.0",
    description="A blazing-fast learned search library for chaotic datasets",
    author="Cynapse ψ∆Ξ",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "joblib>=1.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False
)
