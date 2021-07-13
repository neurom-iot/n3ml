import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='n3ml-python',
    version='0.51.5',
    author='chatterboy',
    author_email='chatter0502@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib==3.3.1',
        'numpy==1.19.1',
        'scikit-learn==0.24.1'
    ]
)
