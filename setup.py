from setuptools import setup

setup(
    name='drifto',
    version='0.1.0a',
    description='Automatic featurization and autoML for event analytics',
    author='Drifto Technologies Inc',
    packages=['drifto', 'drifto.ml'],
    install_requires=['pyarrow',
                      'duckdb',
                      'torch',
                      'pytorch_lightning',
                      'onnx']
)
