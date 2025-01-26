from setuptools import setup, find_packages

setup(
    name='millie',
    version='0.1.0',
    author='River Gleichsner',
    author_email='gleichsnerd@gmail.com',
    description='A Milvus ORM with migration support and more',
    packages=find_packages(),  # Automatically find packages in the millie directory
    install_requires=[
        'click>=8.1.7',
        'pymilvus>=2.3.6',
        'docker>=7.0.0',
        'typeguard>=4.4.1',
        'rich>=13.7.1',
        'python-dotenv',
        # Add other dependencies here
    ],
    extras_require={
        'test': [
            'pytest>=8.0.0',
            'pytest-cov>=6.0.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "millie=millie.cli:cli",
        ],
    },
)