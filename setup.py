import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
                 name="DeriveAlive",
                 version="1.0.1",
                 author="Chen Shi, Stephen Slater, Yue Sun",
                 author_email="cshi@hsph.harvard.edu, stephenslater@college.harvard.edu, yuesun@g.harvard.edu",
                 description="An AutoDifferentiation package",
                 long_description="The software library uses the concept of automatic differentiation to solve differentiation problems in scientific computing. Additional features of an optimization suite and a quadratic spline suite are also listed in this documentation.",
                 long_description_content_type="text/markdown",
                 url="https://github.com/cs207-group19/cs207-FinalProject",
                 packages=setuptools.find_packages(),
                 install_requires =[
       'nose>=1.0.0',
       'numpy==1.15.2',
       'packaging==18.0',
       'pyparsing==2.2.2',
       'scipy==1.1.0',
       'six>=1.11.0',
       'matplotlib>=3.0.1'
        ],
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent",
                              ],
                 )
