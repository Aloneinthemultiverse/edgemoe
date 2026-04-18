from setuptools import setup, find_packages, Extension

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

try:
    from Cython.Build import cythonize
    ext_modules = []
except ImportError:
    ext_modules = []

setup(
    name="edgemoe",
    version="0.1.0",
    author="Sujit Narrayan M",
    author_email="aerospacesujitnarrayan@gmail.com",
    description="Expert-aware MoE inference engine for consumer hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sujitnarrayan/edgemoe",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "edgemoe=edgemoe.cli:main",
        ],
    },
    include_package_data=True,
    package_data={"edgemoe": ["kernels/*.c", "kernels/*.h"]},
)
