import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ParEx',
    packages=['ParEx'],
    version='0.1.0',
    description='Paraphrase using keyword extraction',
    author='Le Minh Khoi',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["ParEx"],
    install_requires=['torch', 'pytorch-lightning', 'wandb', 'transformers', 'datasets', 'tokenizers', 'evaluate', 'sacrebleu', 'sacremoses', 'keybert']
)