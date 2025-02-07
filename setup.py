"""
Setup script
"""

from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.rstrip() for line in f]

setup(
    name="emotiefflib",
    version="0.4.0",
    license="Apache-2.0",
    author="Andrey Savchenko, Egor Churaev",
    author_email="andrey.v.savchenko@gmail.com, egor.churaev@gmail.com",
    packages=find_packages("."),
    download_url="https://github.com/HSE-asavchenko/hsemotion-onnx/archive/v0.3.1.tar.gz",
    url="https://github.com/HSE-asavchenko/face-emotion-recognition",
    description="EmotiEffLib Python Library for Facial Emotion and Engagement Recognition",
    keywords=[
        "face expression recognition",
        "emotion analysis",
        "facial expressions",
        "engagement detection",
    ],
    install_requires=requirements,
)
