# setup.py
from setuptools import setup, find_packages

setup(
    name='rag_system',
    version='0.1.0',  # Or whatever version number you want
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
    "langchain",
    "langchain_openai",
    "langchain_community",
    "python-dotenv"
], #Dependencies are specified here
)