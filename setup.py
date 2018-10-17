import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
                 name="DeriveAlive",
                 version="0.0.1",
                 author="Chen Shi, Stephen Slater, Yue Sun",
                 author_email="cshi@hsph.harvard.edu, stephenslater@college.harvard.edu, yuesun@g.harvard.edu",
                 description="An autodifferentiation package",
                 url="https://github.com/cs207-group19/cs207-FinalProject",
                 packages=setuptools.find_packages(),
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "Operating System :: OS Independent",
                              ],
                 )
