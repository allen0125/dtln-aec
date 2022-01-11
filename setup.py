from setuptools import setup


requires = [
    'tensorflow',
    'wavinfo',
    'librosa',
]

tests_require = ['pytest']

setup(
    name='aec',
    description='DTLN AEC',
    long_description_content_type='text/markdown',
    setup_requires=['setuptools>=38.6.0'],  # long_description_content_type support
    python_requires='>=3.6',  # varialble annotation support
    zip_safe=False,
    install_requires=requires,
    tests_require=tests_require,
)
