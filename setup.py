from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cynsearch",
    version="0.1.0",
    description="A blazing-fast learned search library for chaotic datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cynapse ψ∆Ξ",
    license="MIT",
    url="https://github.com/cyn-apse/cynsearch",
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
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False
)
