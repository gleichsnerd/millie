from setuptools import setup, find_packages

setup(
    name='millie',
    version='0.1.0',
    author='River Gleichsner',
    author_email='gleichsnerd@gmail.com',
    description='A Milvus ORM with migration support and more',
    packages=find_packages(),  # Automatically find packages in the millie directory
    install_requires=[
        'click',
        'pymilvus',
        'python-dotenv',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)