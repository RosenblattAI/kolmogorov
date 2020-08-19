import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setuptools.setup(
    name='generator_scripts',
    version='0.0.1',
    author='Taslim Dosunmu',
    description='Python tools used for satellite situational awareness research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Rosenblatt-LLC/kolmogorov',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    scripts=[
        'bin/distort_dataset',
        'bin/generic_dataset',
        'bin/check_duplicates'
    ]
)
