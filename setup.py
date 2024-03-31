from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name="ihvit",
    version="0.0.1",
    description="inhouse ViT",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    include_package_data=True, # necessary for including data indicated in MANIFEST.in
    entry_points={
        "console_scripts": [
            "ihvit=ihvit.main:main",
            "ihvit.test=ihvit.main:test",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)