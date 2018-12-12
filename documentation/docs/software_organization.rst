Software Organization
=====================

Current directory structure
---------------------------

::

	cs207-FinalProject/
	|-- DeriveAlive/
	|   |-- DeriveAlive.py
	|   |-- __init__.py
	|   |-- optimize.py    
	|   |-- rootfinding.py
	|   `-- spline.py
	|-- demos/
	|   |-- Presentation.ipynb
	|   `-- surprise.py
	|-- documentation/
	|   |-- docs/
	|   |-- documentation.pdf
	|   |-- milestone1.pdf
	|   `-- milestone2.pdf
	|-- tests/
	|   |-- __init__.py
	|   |-- test_DeriveAlive.py
	|   |-- test_optimize.py
	|   |-- test_rootfinding.py
	|   `-- test_spline.py
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

-  ``optimize``: This module utilizes our custom library for
   autodifferentiation to perform optimization. It includes 
   ``DeriveAlive.Var`` class-specific methods. Users can define a custom function to optimize, where this function is :math:`\mathbb{R}^{1} \rightarrow \mathbb{R}^{1}` or :math:`\mathbb{R}^{m} \rightarrow \mathbb{R}^{1}`. If the function is :math:`\mathbb{R}^{m} \rightarrow \mathbb{R}^{1}`, it must take as input a list of :math:`m` variables. Our suggestion is to extract the variables from this list on the first line of the user-defined function, and then use them individually. Furthermore, ``optimize`` allows for dataset compatability with regression optimization. A user can input a numpy matrix with :math:`m` rows and :math:`n` columns, where :math:`n >= 2` and :math:`m >= 1`. The first :math:`n - 1` columns denote the features of the data, and the final column represents the labels. The user must specify the function to optimize as "mse". Then, the function will find a local minimum of the mean squared error objective function. Finally, the module allows for static and animated plots in 2D to 4D using ``plot_results``.

-  ``rootfinding``: This module utilizes our custom library for
   autodifferentiation to find roots of a given :math:`\mathbb{R}^{1} \rightarrow \mathbb{R}^{1}`
   or :math:`\mathbb{R}^{m} \rightarrow \mathbb{R}^{1}` function. It includes 
   ``DeriveAlive.Var`` class-specific methods for Newton's method. It also allows the user to visualize static or animated results in 2D to 4D using ``plot_results``.

-  ``spline``: This module utilizes our custom library for
   autodifferentiation to draw quadratic splines and return corresponding coefficients for quadratic functions of a given scalar function. It includes  ``DeriveAlive.Var`` class-specific methods for quadratic spline generation.

Test Suite
--------------------------------

All test files live in ``tests/`` folder.

-  ``test_DeriveAlive``: This is a test suite for ``DeriveAlive``. It includes tests
   for scalar functions and vector functions to ensure that the ``DeriveAlive`` module
   properly calculates values of scalar functions and gradients with
   respect to scalar inputs, and vector functions and gradients with
   respect to vector inputs.

-  ``test_rootfinding``: This is a test suite for ``rootfinding``.

-  ``test_optimize``: This is a test suite for ``optimization``.

-  ``test_spline``: This is a test suite for ``spline``.

We use Travis CI mfor automatic testing for each push, and Coveralls for 
line coverage metrics. We have already set up these integrations, with
badges included in the ``README.md``. Users may run the test suite by 
navigating to the ``tests/`` folder and running the command ``pytest test_<module>.py``
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
			  >>> from DeriveAlive import DeriveAlive as da
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

