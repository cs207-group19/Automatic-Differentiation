Software Organization
=====================

Current directory structure
---------------------------

::

    cs207-FinalProject/
    |-- DeriveAlive/
    |   |-- DeriveAlive.py
    |   |-- __init__.py
    |   |-- optimization.py    
    |   |-- root_finding.py
    |   `-- spline.py
    |-- demos/
    |   `-- Presentation.ipynb
    |-- documentation/
    |   |-- docs/
    |   |-- milestone1.pdf
    |   `-- milestone2.pdf
    |-- tests/
    |   |-- __init__.py
    |   |-- gradient_descent_demo.py
    |   |-- normalized.txt
    |   |-- test_DeriveAlive.py
    |   `-- test_root_finding.py
    |-- LICENSE
    |-- __init__.py
    |-- README.md
    |-- requirements.txt
    |-- setup.cfg
    `-- setup.py

Basic modules and their functionality
-------------------------------------

-  ``DeriveAlive``: This module contains our custom library for
   autodifferentiation. It includes functionality for a ``Var`` class
   that contains values and derivatives, as well as class-specific
   methods for the operations that our model implements (e.g., tangent,
   sine, power, exponentiation, addition, multiplication, and so on).

-  ``test_DeriveAlive``: This is a test suite for our module
   (explanation in the following section). It currently includes tests
   for scalar functions to ensure that the ``DeriveAlive`` module
   properly calculates values of scalar functions and gradients with
   respect to scalar inputs.

-  ``optimization``: This module utilizes our custom library for
   autodifferentiation to perform optimization. It includes 
   ``DeriveAlive.Var`` class-specific methods for gradient descent.

-  ``root_finding``: This module utilizes our custom library for
   autodifferentiation to find roots of a given :math:`\mathbb{R}^{1} \rightarrow \mathbb{R}^{1}`
   or :math:`\mathbb{R}^{m} \rightarrow \mathbb{R}^{1}` function. It includes 
   ``DeriveAlive.Var`` class-specific methods for Newton's method.

-  ``spline``: This module utilizes our custom library for
   autodifferentiation to draw quadratic splins of a given scalar function.
   It includes  ``DeriveAlive.Var`` class-specific methods for quadratic
   spline generation.

Test Suite
--------------------------------

Our test suite for ``DeriveAlive`` is currently in a test file called
``test_DeriveAlive.py`` in its own ``tests`` folder. 

Our test suite for ``root_finding`` and  ``optimization`` are also in
``tests`` folder, currently in a test file called ``test_root_finding.py``
in the same ``tests`` folder.

We use Travis CI mfor automatic testing for each push, and Coveralls for 
line coverage metrics. We have already set up these integrations, with
badges included in the ``README.md``. Users may run the test suite by 
navigating to the ``tests/`` folder and running the command ``pytest test_DeriveAlive.py``
from the command line (or ``pytest tests`` if the user is outside the
``tests/`` folder).

Installation using PyPI and GitHub
-------------------------------------

We provide two ways for our package installation: PyPI and GitHub.

-  Installation using PyPI

   | We also utilized the Python Package Index (PyPI) for distributing
     our package. PyPI is the official third-party software repository
     for Python and primarily hosts Python packages in the form of
     archives called sdists (source distributions) or precompiled
     wheels. The url to the project is
     https://pypi.org/project/DeriveAlive/.

   -  Create a virtual environment and activate it

      ::

              # If you don't have virtualenv, install it
              sudo easy_install virtualenv
              # Create virtual environment
              virtualenv env
              # Activate your virtual environment
              source env/bin/activate

   -  Install DeriveAlive using pip. In the terminal, type:

      ::

              pip install DeriveAlive

   -  Run module tests before beginning.

      ::

              # Navigate to https://pypi.org/project/DeriveAlive/#files
              # Download tar.gz folder, unzip, and enter the folder
              pytest tests

   -  Use DeriveAlive Python package # (see demo in Section 2.2)

      ::

              python
              >>> import DeriveAlive.DeriveAlive as da
              >>> import numpy as np
              >>> x = da.Var([np.pi/2])
              >>> x
              Var([1.57079633], [1.])
              ...
              >>> quit()

              # deactivate virtual environment
              deactivate

-  Installation from GitHub

   -  Download the package from GitHub to your folder via these commands
      in the terminal:

      ::

              mkdir test_cs207
              cd test_cs207/
              git clone https://github.com/cs207-group19/cs207-FinalProject.git
              cd cs207-FinalProject/

   -  Create a virtual environment and activate it

      ::

              # If you don't have virtualenv, install it
              sudo easy_install virtualenv
              # Create virtual environment
              virtualenv env
              # Activate your virtual environment
              source env/bin/activate


   -  Install required packages and run module tests in ``tests/``

      ::

              pip install -r requirements.txt
              pytest tests

   -  Use DeriveAlive Python package (see demo in Section 2.2)

      ::

              python
              >>> import DeriveAlive.DeriveAlive as da
              >>> import numpy as np
              >>> x = da.Var([np.pi/2])
              >>> x
              Var([1.57079633], [1.])
              ...
              >>> quit()

              # deactivate virtual environment
              deactivate

