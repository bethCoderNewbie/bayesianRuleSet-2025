=================
Bayesian Rule Set
=================

A Python implementation of the Bayesian Rule Set algorithm for interpretable classification as described in Wang et al. (2016).

Overview
--------

Bayesian Rule Set is an interpretable machine learning algorithm that discovers a set of simple rules for binary classification. It uses Bayesian inference and MCMC sampling to find the optimal set of rules that balance accuracy and interpretability.

The algorithm has the following key features:

* Generates human-readable IF-THEN rules
* Balances accuracy and rule set complexity through Bayesian optimization
* Handles mixed data types (categorical and numerical features)
* Provides interpretable outputs for decision-making

Installation
-----------

You can install the package directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/yourusername/bayesian-rule-set.git

Or install in development mode:

.. code-block:: bash

    git clone https://github.com/yourusername/bayesian-rule-set.git
    cd bayesian-rule-set
    pip install -e .

Dependencies
-----------

* numpy
* pandas
* scipy
* scikit-learn
* matplotlib

Usage
-----

Basic usage example:

.. code-block:: python

    from ruleset import BayesianRuleSet
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data
    df = pd.read_csv('data/coupon_data.csv')
    X = df.drop('Y', axis=1)
    y = df['Y']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = BayesianRuleSet(max_rules=5000, max_iter=50000, support=3, maxlen=3, method='forest')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print the learned rules
    model.print_rules(model.predicted_rules)

For a more detailed example, see the scripts in the `examples` directory.

Reference
---------

Wang, T., Rudin, C., Doshi-Velez, F., Liu, Y., Klampfl, E., & MacNeille, P. (2016).
"Bayesian Rule Sets for Interpretable Classification."
IEEE 16th International Conference on Data Mining (ICDM).

License
-------

This project is licensed under the MIT License - see the LICENSE.txt file for details.