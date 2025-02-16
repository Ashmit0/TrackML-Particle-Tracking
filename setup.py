from setuptools import setup, find_packages

setup(
    name="Track-ML-Package",  # Package name
    version="0.1.0",       # Package version
    author="Ashmit Bathla",
    author_email="ashmitb210216@gmail.com",
    description="Machine Learning Module implementing MetaLayer and ParticleNet desigend for TrackML data.",
    long_description=open("README.md").read(),  # Use README.md as the long description
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/my_ml_package",  # Update with your repo URL

    # Automatically find packages in subdirectories
    packages=find_packages(),

    # Dependencies (same as those in requirements.txt)
    install_requires=[
        "numpy", 
        "pandas", 
        "matplotlib", 
        "seaborn", 
        "vector", 
        "awkward", 
        "torch", 
        "torchvision", 
        "torchaudio",
        "torch_geometric"
    ],

    # Classifiers help categorize your package
    classifiers=[
        "Programming Language :: Python :: 3.10.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence and Application to High Energy Physics.",
    ],

    python_requires=">=3.10",  # Specify compatible Python versions
    include_package_data=False,  # Include non-code files (e.g., configs, data files)
    zip_safe=True,  # False means it can be installed as an editable package
)