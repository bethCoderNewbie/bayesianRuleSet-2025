# coding=utf-8
"""
Bayesian Rule Set Mining Implementation

This module implements the Bayesian Rule Set algorithm for interpretable classification
as described in Wang et al. (2016). It discovers a set of simple rules for binary
classification while maintaining interpretability.

Reference:
    Wang, T., Rudin, C., Doshi-Velez, F., Liu, Y., Klampfl, E., & MacNeille, P. (2016).
    "Bayesian Rule Sets for Interpretable Classification."
    IEEE 16th International Conference on Data Mining (ICDM).

Original Authors (Concept & Initial Implementation):
    Tong Wang
    Peter (Zhen) Li

Modifications/Refinements (based on provided Colab script):
    User (As provided in the Colab script)

License: MIT
"""
from __future__ import annotations # For type hinting forward references if needed

import numpy as np
import pandas as pd
import math
import re
import itertools
from bisect import bisect_left
from random import sample, random
import time
import operator
from collections import Counter, defaultdict
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, f1_score
import matplotlib.pyplot as plt
import warnings

# Ignore specific warnings if desired (e.g., RuntimeWarning for division by zero)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Helper Functions ---

def accumulate(iterable, func=operator.add):
    """Accumulate results of function applications.

    Args:
        iterable: Input iterable
        func: Function to apply (default: addition)

    Yields:
        Running totals
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """Find rightmost value less than x in sorted array a."""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        # Changed to handle edge case where x is smaller than all elements
        return 0

def remove_duplicates(l):
    """Remove duplicate lists while preserving order within each list."""
    return [list(x) for x in set(tuple(x) for x in l)]

def find_interval(idx1, l2):
    """Find interval index for a given value based on cumulative lengths."""
    idx2 = 0
    tmp_sum = 0
    for i in l2:
        tmp_sum += i
        if tmp_sum > idx1: # Use > instead of >=
            return idx2
        else:
            idx2 += 1
    return idx2 # Return last index if not found earlier

def log_betabin(k, n, alpha, beta):
    """Calculate the log of the Beta-Binomial probability.

    Args:
        k (int or float): Number of successes
        n (int or float): Number of trials
        alpha (float): First parameter of the Beta distribution (prior successes + 1)
        beta (float): Second parameter of the Beta distribution (prior failures + 1)

    Returns:
        float: Log of the Beta-Binomial probability, or large negative number on error.
    """
    # Ensure inputs are non-negative and k <= n
    k = max(0, k)
    n = max(0, n)
    if k > n:
       # This case shouldn't happen with TP/FP/TN/FN, but handle defensively
       k = n

    # Add small epsilon to avoid log(0) or lgamma(0) issues
    eps = 1e-9
    alpha = max(eps, alpha)
    beta = max(eps, beta)
    # Ensure arguments to lgamma are positive
    k_plus_alpha = max(eps, k + alpha)
    n_minus_k_plus_beta = max(eps, n - k + beta)
    n_plus_alpha_beta = max(eps, n + alpha + beta)
    alpha_plus_beta = max(eps, alpha + beta)

    try:
        # Use gammaln for log gamma calculation
        c = math.lgamma(alpha_plus_beta) - math.lgamma(alpha) - math.lgamma(beta)
        log_beta_val = (math.lgamma(k_plus_alpha) +
                        math.lgamma(n_minus_k_plus_beta) -
                        math.lgamma(n_plus_alpha_beta))
        return log_beta_val + c
    except ValueError:
        # Handle potential domain errors in lgamma if inputs are still problematic
        print(f"ValueError in log_betabin: k={k}, n={n}, alpha={alpha}, beta={beta}")
        # Return a very small number instead of NaN or error
        return -1e9 # Return a large negative number (log scale)

def get_confusion(yhat, y):
    """Calculate confusion matrix elements (TP, FP, TN, FN)."""
    yhat = np.asarray(yhat).astype(bool)
    y = np.asarray(y).astype(bool)
    if len(yhat) != len(y):
        raise ValueError(f'yhat ({len(yhat)}) has different length than y ({len(y)})')
    TP = np.sum(yhat & y)
    FP = np.sum(yhat & ~y)
    TN = np.sum(~yhat & ~y)
    FN = np.sum(~yhat & y)
    # Ensure consistency check (optional, can slow down computation)
    # assert TP + FP + TN + FN == len(y), f"Confusion matrix counts error: {TP}+{FP}+{TN}+{FN} != {len(y)}"
    return TP, FP, TN, FN

def extract_rules(tree, feature_names):
    """Extract rules from a fitted decision tree as lists of feature indices.

    Args:
        tree: A fitted sklearn DecisionTree object.
        feature_names: A list of identifiers (e.g., integer indices starting from 1)
                       corresponding to the columns used to fit the tree.

    Returns:
        list: A list of rules, where each rule is a list of feature indices (int).
              Returns an empty list if the tree is trivial or uses no valid features.
    """
    left = tree.tree_.children_left
    # Handle trivial tree (just a leaf node)
    if left[0] == -1:
        return []
    right = tree.tree_.children_right

    # Filter feature indices used in the tree to be within bounds of provided names
    # Note: feature_names here are expected to be the indices (1-based) used internally
    valid_feature_indices_in_tree = [i for i in tree.tree_.feature if 0 <= i < len(feature_names)]
    if not valid_feature_indices_in_tree:
         return [] # No valid features used in the tree

    # Get the indices of the leaf nodes
    leaf_node_indices = np.where(left == -1)[0]

    def find_path(node_id, current_path=None):
        """Recursive helper to find the path from root to a leaf."""
        if current_path is None:
            current_path = []

        # Find parent node and the decision that led to the current node
        # Search in left children
        parent_indices_left = np.where(left == node_id)[0]
        if len(parent_indices_left) > 0:
            parent_id = parent_indices_left[0]
            feature_idx_at_parent = tree.tree_.feature[parent_id]
             # Decision was feature <= threshold (represented by the feature index itself)
            # Ensure feature index is valid before adding
            if 0 <= feature_idx_at_parent < len(feature_names):
                 # Use the actual identifier from feature_names
                current_path.append(feature_names[feature_idx_at_parent])
            else:
                # Invalid feature index encountered in path, discard this path
                return None
        # Search in right children
        else:
            parent_indices_right = np.where(right == node_id)[0]
            if len(parent_indices_right) > 0:
                parent_id = parent_indices_right[0]
                # Decision was feature > threshold (we don't explicitly store this,
                # the absence of the feature in the left path implies it)
            else:
                 # Node not found as child (should not happen for nodes > 0 in valid tree)
                 # If node_id is 0 (root), we stop recursion.
                 if node_id == 0:
                     current_path.reverse() # Path is built bottom-up
                     return current_path
                 else:
                    # Problem finding parent for a non-root node
                    return None # Indicate an invalid path

        # Recursively find the rest of the path towards the root
        if parent_id == 0:
             current_path.reverse() # Path is built bottom-up
             return current_path
        else:
            # Check parent validity before recursion
            if parent_id >= len(left) or parent_id < 0: return None
            return find_path(parent_id, current_path)

    rules = []
    for leaf_id in leaf_node_indices:
        # Check leaf validity
        if leaf_id >= len(left) or leaf_id < 0: continue

        # Find the path (sequence of feature indices) leading to this leaf
        path_indices = find_path(leaf_id)

        # Add the rule if a valid path was found
        if path_indices is not None and isinstance(path_indices, list):
            # Ensure rule contains only valid integer indices and is not empty
            valid_rule = [item for item in path_indices if isinstance(item, (int, np.integer))]
            if valid_rule:
                rules.append(sorted(valid_rule)) # Sort for consistency

    # Filter out empty or potentially invalid rules one last time
    # And remove duplicates
    unique_rules = [list(r) for r in set(tuple(r) for r in rules if r)]
    return unique_rules


# --- Main BayesianRuleSet Class ---

class BayesianRuleSet:
    """Implementation of the Bayesian Rule Set algorithm.

    Based on "A Bayesian Framework for Learning Rule Sets for Interpretable Classification"
    by Tong Wang, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, Perry MacNeille (2016).

    This version incorporates modifications and parameter tuning from the provided Colab script.

    Args:
        max_rules (int): Max rules to keep after screening (if support generates more). Default 5000.
        maxlen (int): Max conditions per rule. Default 3.
        support (float): Min % of POSITIVE class samples a rule must cover. Default 3.
        max_iter (int): MCMC iterations. Default 50000.
        chains (int): Number of MCMC chains (currently only 1 chain is implemented). Default 1.
        alpha_1 (float): Beta prior parameter (alpha) for P(Pos | Rule). Default 20.
        beta_1 (float): Beta prior parameter (beta) for P(Pos | Rule). Default 1.
        alpha_2 (float): Beta prior parameter (alpha) for P(Neg | ~Rule). Default 20.
        beta_2 (float): Beta prior parameter (beta) for P(Neg | ~Rule). Default 1.
        alpha_l (list[float], optional): Beta prior parameter (alpha) for rule length probability P(l). Defaults to all 1s.
        beta_l (list[float], optional): Beta prior parameter (beta) for rule length probability P(l). Defaults based on pattern space.
        level (int): Number of quantiles for discretizing numeric features. Default 4.
        neg (bool): Whether to allow negated conditions for categorical features. Default True.
        add_rules (list): User-defined rules (as lists of item indices) to add to the pool. Default [].
        criteria (str): Criteria ('precision' or 'information') for selecting rules if `max_rules` is exceeded during screening. Default 'information'.
        greedy_initilization (bool): Whether to use a greedy approach for the initial ruleset in MCMC. Default False.
        greedy_threshold (float): Threshold used in the (currently simplified) greedy initialization. Default 0.05.
        propose_threshold (float): Probability of exploration vs. exploitation in MCMC proposal. Default 0.3.
        method (str): Method for initial rule generation ('forest' or 'fpgrowth'). Default 'forest'.
        forest_size (int): Number of trees (base estimator count) for 'forest' method. Default 500.
        rule_adjust (bool): Whether to perform rule adjustment for numeric cutoffs (currently simplified). Default True.
        binary_input (bool): If True, assumes input X is already binarized and skips transformation. Default False.

    Attributes:
        rules (list): List of all candidate rules generated and screened. Each rule is a list of integer item IDs.
        rules_len (list): List of lengths corresponding to the rules in `self.rules`.
        itemNames (dict): Mapping from integer item ID to human-readable feature condition string (e.g., {1: 'Age<25.5', 2: 'Gender_Male'}).
        predicted_rules (list): List of integer indices (referring to `self.rules`) selected for the final model.
        rule_explainations (dict): Stores potentially adjusted rule meanings and coverage vectors after `modify_rule`.
        attributeNames (pd.Index): Original column names of the input DataFrame X.
        cutoffs (dict): Stores calculated cutoffs for numerical features.
        supp (np.ndarray): Support counts for each rule in `self.rules`.
        maps (defaultdict): Stores history of MCMC states (scores, rulesets) during training.
    """

    def __init__(self, max_rules=5000, max_iter=50000, chains=1,
                 support=3, maxlen=3, alpha_1=20, beta_1=1,
                 alpha_2=20, beta_2=1, alpha_l=None,
                 beta_l=None, level=4, neg=True,
                 add_rules=[], criteria='information',
                 greedy_initilization=False, greedy_threshold=0.05,
                 propose_threshold=0.3,
                 method='forest', forest_size=500,
                 rule_adjust=True, binary_input=False):

        if not isinstance(maxlen, int) or maxlen < 1:
            raise ValueError("maxlen must be a positive integer.")
        if not isinstance(max_rules, int) or max_rules < 1:
            raise ValueError("max_rules must be a positive integer.")
        if not isinstance(support, (int, float)) or not (0 <= support <= 100):
             raise ValueError("support must be a percentage between 0 and 100.")
        if not isinstance(max_iter, int) or max_iter < 1:
             raise ValueError("max_iter must be a positive integer.")
        if method not in ['forest', 'fpgrowth']:
            raise ValueError("method must be 'forest' or 'fpgrowth'.")

        self.max_rules = max_rules
        self.max_iter = max_iter
        self.chains = chains # Note: Multi-chain execution is not implemented in this version
        self.support = support # Support as percentage of positive class
        self.maxlen = maxlen
        self.alpha_1 = alpha_1 # Positive Likelihood Alpha (TP / (TP + FP))
        self.beta_1 = beta_1  # Positive Likelihood Beta
        self.alpha_2 = alpha_2 # Negative Likelihood Alpha (TN / (TN + FN))
        self.beta_2 = beta_2  # Negative Likelihood Beta
        self.alpha_l = alpha_l # Prior Alpha for rule length l
        self.beta_l = beta_l   # Prior Beta for rule length l
        self.level = level     # Discretization levels for numeric features
        self.neg = neg         # Allow negated categorical features
        self.add_rules = add_rules # User-provided rules
        self.criteria = criteria   # Rule screening criterion
        self.greedy_initilization = greedy_initilization # Use greedy start for MCMC?
        self.greedy_threshold = greedy_threshold
        self.propose_threshold = propose_threshold # MCMC proposal exploration probability
        self.method = method       # Rule generation method
        self.forest_size = forest_size # Number of base trees for forest method
        self.rule_adjust = rule_adjust # Adjust numeric rule cutoffs? (Currently simplified)
        self.binary_input = binary_input # Is input data already binarized?

        # Internal state attributes initialized during fitting
        self.rules = []
        self.rules_len = []
        self.itemNames = dict()
        self.predicted_rules = []
        self.rule_explainations = dict()
        self.attributeNames = None
        self.cutoffs = dict()
        self.supp = np.array([])
        self.maps = defaultdict(list) # Store MCMC results (chain 0)
        self.patternSpace = None
        self.Lup = None
        self.P0 = None
        self.const_denominator = None
        self.yhat = None # Stores last prediction vector during probability computation
        self.Asize = [] # MCMC bounds (for potential optimization)
        self.C = []     # MCMC bounds (for potential optimization)


    def transform(self, df: pd.DataFrame, neg: bool = True) -> pd.DataFrame:
        """Transforms input DataFrame with mixed types into a binary DataFrame.

        Categorical features are one-hot encoded (with optional negation).
        Numerical features are discretized based on quantiles.

        Args:
            df (pd.DataFrame): Input data with original features.
            neg (bool): If True, create negated versions of categorical features
                        (e.g., 'feature != value').

        Returns:
            pd.DataFrame: Binarized DataFrame where columns represent conditions.

        Raises:
            ValueError: If input is not a Pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
             raise ValueError("Input 'df' must be a pandas DataFrame.")

        self.attributeNames = df.columns
        level = self.level
        df_cat = pd.DataFrame()
        df_num = pd.DataFrame()

        # Separate categorical and numerical columns
        for col in df.columns:
            # Check if column name is problematic (contains reserved chars or is digit)
            # This check should ideally happen *before* calling transform if possible.
            if any(c in str(col) for c in ['_', '<']) or str(col).isdigit():
                warnings.warn(f"Column name '{col}' contains reserved characters ('_', '<') or is purely numeric. "
                              "This might cause issues. Consider renaming columns beforehand.")

            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                df_cat[col] = df[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                df_num[col] = df[col]
            # else: ignore other types like datetime etc.

        df_cat_processed = pd.DataFrame(index=df.index) # Ensure index alignment
        # Process categorical columns
        for col in df_cat.columns:
            # Convert to string to handle potential mixed types within object columns
            df_cat[col] = df_cat[col].astype(str)
            items = df_cat[col].unique()

            if len(items) == 1: # Skip columns with only one value
                warnings.warn(f"Categorical column '{col}' has only one unique value ('{items[0]}'). Skipping.")
                continue

            if len(items) == 2:
                # Create only one binary column for 2 categories to avoid redundancy
                # Use the first item value for the positive condition name
                item0_name = f"{col}_{items[0]}"
                df_cat_processed[item0_name] = (df_cat[col] == items[0])
                # Optionally, could add the negation explicitly if neg=True, but it's redundant
                # if neg:
                #    item1_name_neg = f"{col}_{items[0]}_neg" # or f"{col}_{items[1]}"
                #    df_cat_processed[item1_name_neg] = (df_cat[col] != items[0])
            elif len(items) > 2:
                for item in items:
                    item_name = f"{col}_{item}"
                    df_cat_processed[item_name] = (df_cat[col] == item)
                    if neg:
                        # Add negated feature
                        item_name_neg = f"{col}_{item}_neg"
                        df_cat_processed[item_name_neg] = (df_cat[col] != item)

        df_num_processed = pd.DataFrame(index=df.index) # Ensure index alignment
        # Process numerical columns
        for col in df_num.columns:
            # Calculate unique quantiles safely
            try:
                # Use np.linspace for potentially more evenly spaced quantiles
                q_values = np.linspace(0, 1, level + 1)
                quantiles = df_num[col].quantile(q_values).unique()
                quantiles = np.sort(quantiles)
            except Exception as e:
                warnings.warn(f"Could not compute quantiles for column '{col}'. Skipping. Error: {e}")
                continue

            if len(quantiles) < 2: # Not enough variation to create bins
                warnings.warn(f"Numerical column '{col}' has < 2 unique quantile values. Skipping.")
                continue

            # Create conditions based on intervals between unique quantiles
            # Ensure smallest value gets included in the first bin if necessary
            if df_num[col].min() < quantiles[0]:
                 col_name = f"{col}<{quantiles[0]:.4g}" # Use .4g for cleaner formatting
                 df_num_processed[col_name] = (df_num[col] < quantiles[0])

            for i in range(len(quantiles) - 1):
                 lower_bound = quantiles[i]
                 upper_bound = quantiles[i+1]

                 # Skip if lower and upper bound are the same (can happen with skewed data)
                 if np.isclose(lower_bound, upper_bound):
                      continue

                 # Create condition: lower_bound <= feature < upper_bound
                 # We represent this with two binary features:
                 # 1. feature >= lower_bound (value < feature format)
                 col_name_ge = f"{lower_bound:.4g}<{col}"
                 df_num_processed[col_name_ge] = (df_num[col] >= lower_bound)

                 # 2. feature < upper_bound (feature < value format)
                 col_name_lt = f"{col}<{upper_bound:.4g}"
                 df_num_processed[col_name_lt] = (df_num[col] < upper_bound)

            # Ensure largest value gets included in the last bin if necessary
            if df_num[col].max() >= quantiles[-1]:
                 col_name = f"{quantiles[-1]:.4g}<{col}" # Represents feature >= last_quantile
                 df_num_processed[col_name] = (df_num[col] >= quantiles[-1])


        # Combine processed categorical and numerical data
        result = pd.concat([df_cat_processed, df_num_processed], axis=1).astype(bool)

        # Ensure no duplicate column names resulted from the process
        result = result.loc[:, ~result.columns.duplicated()]
        return result


    def set_parameters(self, X_trans: pd.DataFrame):
        """Sets prior parameters based on the transformed feature space."""
        if not isinstance(X_trans, pd.DataFrame):
            raise ValueError("Input X_trans must be a pandas DataFrame.")
        if X_trans.shape[1] == 0:
            raise ValueError("Transformed data X_trans has no columns.")

        numAttributes = X_trans.shape[1]
        self.patternSpace = np.zeros(self.maxlen + 1) # Use zeros for initialization

        # Estimate the size of the pattern space (|A_l|) for each length l
        for i in range(1, self.maxlen + 1):
            # Calculate combinations C(numAttributes, i)
            if numAttributes >= i:
                try:
                    # Use math.comb for accurate calculation, handles large numbers
                    nCr = math.comb(numAttributes, i)
                    # The original paper's calculation is approximate.
                    # A simple upper bound assuming each attribute could be included
                    # or not (2^i) might be too loose. Using nCr is more direct.
                    # We'll stick to an estimate closer to the original script's intent:
                    # Assume each chosen attribute can be either positive or negative form
                    # This isn't strictly true due to discretization but serves as prior size.
                    # A tighter bound would consider the exact structure of transformed features.
                    self.patternSpace[i] = nCr # Start with combinations
                    # Optional: Multiply by 2^i as a rough estimate if considering directions?
                    # self.patternSpace[i] = nCr * (2**i)
                except ValueError: # Handle potential overflow for very large numbers
                    # Fallback to a large number or np.inf if math.comb fails
                    self.patternSpace[i] = np.inf # Or a very large float
                    warnings.warn(f"Calculation of C({numAttributes}, {i}) resulted in overflow. "
                                  f"Using infinity for patternSpace[{i}]. Priors might be affected.")
            else:
                self.patternSpace[i] = 0 # Cannot form rule of length i

        # Handle cases where pattern space is effectively infinite or zero
        self.patternSpace = np.nan_to_num(self.patternSpace, nan=0.0, posinf=1e18) # Replace inf with large number
        self.patternSpace[0] = 1 # Length 0 has 1 pattern (empty rule) - though not typically used in priors

        # Set default priors for rule length if not provided
        if self.alpha_l is None:
            self.alpha_l = np.ones(self.maxlen + 1)
        elif len(self.alpha_l) != self.maxlen + 1:
            raise ValueError(f"alpha_l must have length maxlen+1 ({self.maxlen+1})")

        if self.beta_l is None:
            # Default beta_l encourages sparsity by making it proportional to patternSpace size
            # Add 1 to avoid beta=0 issues if patternSpace is 0
            self.beta_l = np.array([(max(1.0, self.patternSpace[i])) for i in range(self.maxlen + 1)])
            # Consider adding a factor like the original script's *100? (e.g., *10 or *100)
            # self.beta_l = np.array([(max(1.0, self.patternSpace[i]*10 + 1)) for i in range(self.maxlen + 1)])
        elif len(self.beta_l) != self.maxlen + 1:
            raise ValueError(f"beta_l must have length maxlen+1 ({self.maxlen+1})")

        # Ensure alpha/beta are numpy arrays for potential vectorization
        self.alpha_l = np.asarray(self.alpha_l)
        self.beta_l = np.asarray(self.beta_l)

    def convert_y(self, y) -> np.ndarray:
        """Converts the target variable y to a binary (0/1) numpy array."""
        if isinstance(y, pd.Series):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)

        # Check for common string representations of positive class
        positive_labels = {'Y', 'yes', '1', 'true', 'True', 1, True}
        negative_labels = {'N', 'no', '0', 'false', 'False', 0, False}

        unique_vals = set(np.unique(y))

        if unique_vals.issubset({0, 1}):
            return y.astype(int)
        elif unique_vals.issubset({False, True}):
             return y.astype(int)
        elif unique_vals.issubset(positive_labels.union(negative_labels)):
             # Convert based on known labels
             is_positive = np.array([val in positive_labels for val in y])
             is_negative = np.array([val in negative_labels for val in y])
             if np.all(is_positive | is_negative): # Ensure all values are covered
                 return is_positive.astype(int)

        # If only two unique values, try inferring 0 and 1
        if len(unique_vals) == 2:
            val_list = sorted(list(unique_vals))
            warnings.warn(f"Target variable has unique values {unique_vals}. "
                          f"Assuming '{val_list[0]}' maps to 0 and '{val_list[1]}' maps to 1.")
            mapping = {val_list[0]: 0, val_list[1]: 1}
            return np.vectorize(mapping.get)(y)

        raise ValueError(f"Target variable 'y' is not binary or easily convertible. Unique values found: {unique_vals}")

    def precompute(self, y: np.ndarray):
        """Precomputes values needed for probability calculations."""
        if not isinstance(y, np.ndarray):
            y = self.convert_y(y) # Ensure y is numpy array of 0s and 1s

        N = len(y)
        if N == 0:
            raise ValueError("Input target array 'y' is empty.")
        N_pos = np.sum(y)
        N_neg = N - N_pos

        # Calculate Lup: Log-likelihood upper bound (log P(S|A_max))
        # Assumes a hypothetical perfect rule predicting all positives correctly (TP=N_pos, FP=0)
        # and another predicting all negatives correctly (TN=N_neg, FN=0).
        # This seems conceptually different from paper's formula 6, which is likelihood of data given ruleset.
        # Sticking to the Colab script's implementation here.
        TP_max, FP_max, TN_max, FN_max = N_pos, 0, N_neg, 0
        self.Lup = (log_betabin(TP_max, TP_max + FP_max, self.alpha_1, self.beta_1) +
                    log_betabin(TN_max, FN_max + TN_max, self.alpha_2, self.beta_2))

        # Precompute constant part of the prior P(M_l=0 | A_l) = BetaBin(0 | |A_l|, alpha_l, beta_l)
        # This is the prior probability of *not* selecting any rules of length l.
        # P0 is the sum of these log-priors across all lengths (product in normal probability space).
        # Represents the prior probability of the empty ruleset.
        Kn_count_empty = np.zeros(self.maxlen + 1, dtype=int)
        self.P0 = 0.0
        for i in range(1, self.maxlen + 1):
            if self.patternSpace[i] > 0: # Only compute if pattern space exists
                self.P0 += log_betabin(Kn_count_empty[i], self.patternSpace[i],
                                       self.alpha_l[i], self.beta_l[i])

        # Precompute the denominator constant for bound updates (Formula 9 in paper)
        self.const_denominator = np.full(self.maxlen + 1, -np.inf) # Initialize with log(0)
        for i in range(1, self.maxlen + 1):
            # Ensure inputs are valid and avoid division by zero or log(<=0)
            ps_i = self.patternSpace[i]
            al_i = self.alpha_l[i]
            be_i = self.beta_l[i]
            numerator = ps_i + be_i - 1
            denominator = ps_i + al_i - 1

            if ps_i > 0 and numerator > 0 and denominator > 0:
                # Use np.log for safe handling of potential precision issues
                self.const_denominator[i] = np.log(numerator / denominator)
            # else: remains -inf


    def compute_cutoffs(self, X: pd.DataFrame, y: np.ndarray):
        """Computes potential cutoffs for numerical features based on class changes."""
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame.")
        if not isinstance(y, np.ndarray):
             y = self.convert_y(y) # Ensure y is numpy array

        self.cutoffs = dict()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Drop NaNs before processing
                valid_idx = X[col].notna() & pd.notna(y) # Ensure y is also not NaN
                if not np.any(valid_idx):
                    warnings.warn(f"Column '{col}' or corresponding 'y' contains all NaNs. Skipping cutoffs.")
                    continue

                col_vals = X.loc[valid_idx, col].to_numpy()
                y_vals = y[valid_idx]

                if len(np.unique(col_vals)) < 2: # Not enough variation
                     continue

                # Sort values by feature value
                sorted_indices = np.argsort(col_vals)
                tmps = list(zip(col_vals[sorted_indices], y_vals[sorted_indices]))

                cutoff_points = set()
                # Estimate interval based on range and levels for modification step
                data_range = col_vals.max() - col_vals.min()
                interval = data_range / self.level if data_range > 0 and self.level > 0 else 1.0

                for i in range(len(tmps) - 1):
                    val1, class1 = tmps[i]
                    val2, class2 = tmps[i+1]
                    # Add cutoff if class label changes and values are different
                    if class1 != class2 and not np.isclose(val1, val2):
                        # Use midpoint as cutoff
                        midpoint = (val1 + val2) / 2.0
                        cutoff_points.add(midpoint)

                # Store unique, sorted cutoffs and the estimated interval
                self.cutoffs[col] = (sorted(list(cutoff_points)), interval)


    def generate_rules(self, X_trans: pd.DataFrame, y: np.ndarray):
        """Generates candidate rules using the specified method."""
        print(f"Generating rules using method: {self.method}")
        if not isinstance(X_trans, pd.DataFrame) or not isinstance(y, np.ndarray):
             raise TypeError("X_trans must be a DataFrame and y must be a NumPy array.")

        num_features = X_trans.shape[1]
        if num_features == 0:
            warnings.warn("X_trans has no columns, cannot generate rules.")
            self.rules = []
            self.itemNames = {}
            return

        # Create item names mapping (1-based index to column name)
        self.itemNames = {i + 1: col for i, col in enumerate(X_trans.columns)}
        # Create feature indices list (1-based) for rule extraction
        feature_indices = list(range(1, num_features + 1))

        if self.method == 'fpgrowth':
            try:
                from fim import fpgrowth # Local import
                print("Using fpgrowth...")
                # Prepare data for fpgrowth: list of lists, only positive examples
                itemMatrix = (X_trans.values * feature_indices).astype(int) # Multiply by 1-based index
                itemMatrix_pos = [row[row > 0].tolist() for row in itemMatrix[y == 1]]
                # Ensure items are hashable (standard ints)
                itemMatrix_pos_hashable = [[int(item) for item in sublist] for sublist in itemMatrix_pos]

                # Run fpgrowth
                # Support is given as percentage of *positive* examples
                min_support_count = (self.support / 100.0) * np.sum(y==1)
                # FIM support is often absolute or relative to total N. Adjust if needed.
                # Assuming FIM support is relative count / total positive count
                fim_support_threshold = self.support # Pass percentage directly

                rules_raw = fpgrowth(itemMatrix_pos_hashable,
                                     supp=fim_support_threshold, # Support threshold (%)
                                     zmin=1,           # Min rule length
                                     zmax=self.maxlen, # Max rule length
                                     report='S')       # Report support counts alongside rules

                # Extract rule items (first element of tuple) and sort
                generated_rules = [sorted(list(rule[0])) for rule in rules_raw]
                print(f"fpgrowth found {len(generated_rules)} rules initially.")
                self.rules = generated_rules

            except ImportError:
                warnings.warn("fpgrowth method requires the 'fim' package (e.g., pyfim). "
                              "Falling back to 'forest' method.")
                self.method = 'forest' # Fallback if fim not installed
            except Exception as e:
                warnings.warn(f"Error during fpgrowth: {e}. Falling back to 'forest' method.")
                self.method = 'forest' # Fallback on other errors

        if self.method == 'forest':
            print("Using forest method...")
            rules_set = set() # Use set for efficient duplicate removal
            total_trees_evaluated = 0

            # Train forests of different depths to get rules of varying lengths
            for length in range(1, self.maxlen + 1):
                # Determine number of trees, ensuring it's positive
                # Scale forest size by length? Original script did this.
                n_trees = max(1, self.forest_size * length)
                print(f" Training forest for max_depth={length} with {n_trees} estimators...")

                # Use balanced class weight for imbalanced datasets
                clf = RandomForestClassifier(n_estimators=n_trees,
                                             max_depth=length,
                                             random_state=42, # For reproducibility
                                             class_weight='balanced',
                                             min_samples_leaf=max(2, int(0.005 * len(y))), # Avoid overfitting tiny leaves
                                             n_jobs=-1) # Use all available cores
                clf.fit(X_trans, y)
                total_trees_evaluated += len(clf.estimators_)

                # Extract rules from each tree in the forest
                for estimator in clf.estimators_:
                    # Pass the 1-based feature indices to extract_rules
                    tree_rules = extract_rules(estimator, feature_indices)
                    for rule in tree_rules:
                        # Add non-empty rules (as sorted tuples) to the set
                        if rule:
                             rules_set.add(tuple(rule)) # Already sorted by extract_rules

            # Convert unique rule tuples back to lists
            self.rules = [list(r) for r in rules_set]
            print(f"Forest method extracted {len(self.rules)} unique rules from {total_trees_evaluated} trees.")

        # Add user-defined rules (ensure they are valid indices)
        num_added_user = 0
        if self.add_rules:
            print(f"Adding {len(self.add_rules)} user-defined rules...")
            existing_rules_set = set(tuple(r) for r in self.rules)
            for add_rule in self.add_rules:
                # Validate the rule contents
                if isinstance(add_rule, list) and all(isinstance(item, int) and 1 <= item <= num_features for item in add_rule):
                    sorted_rule_tuple = tuple(sorted(add_rule))
                    if sorted_rule_tuple not in existing_rules_set:
                        self.rules.append(sorted(add_rule))
                        existing_rules_set.add(sorted_rule_tuple)
                        num_added_user += 1
                else:
                    warnings.warn(f"Skipping invalid user-added rule: {add_rule}. "
                                  f"Rule must be a list of valid integer item indices (1 to {num_features}).")
            print(f" Added {num_added_user} new unique user-defined rules.")

        # Store rule lengths (needed for priors and potentially other steps)
        self.rules_len = [len(rule) for rule in self.rules]
        print(f"Total rules generated: {len(self.rules)}")


    def screen_rules(self, X_trans: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Screens generated rules based on support and max_rules criteria."""
        print("Screening rules...")
        if not self.rules:
             warnings.warn("No rules available for screening.")
             self.supp = np.array([])
             return np.zeros((X_trans.shape[0], 0), dtype=bool)

        y = self.convert_y(y) # Ensure y is binary numpy array
        n_samples = X_trans.shape[0]
        n_positive = np.sum(y)
        if n_positive == 0:
            warnings.warn("No positive samples in 'y' for screening. Support threshold cannot be applied effectively.")
            # Default to a small absolute count if n_positive is 0? Or skip support check?
            min_support_count = 1 # Require at least 1 sample covered
        else:
            # Support threshold is % of positive class samples
            min_support_count = math.ceil((self.support / 100.0) * n_positive)

        print(f" Positive samples: {n_positive}. Min support count required: {min_support_count}")

        # --- Build Rule Matrix Efficiently (Sparse) ---
        rows, cols, data = [], [], []
        valid_rule_indices_map = {} # Map original index to new index after filtering
        new_rule_idx = 0
        screened_rules_temp = []
        screened_rules_len_temp = []

        # Convert X_trans to CSC sparse format for faster column slicing if not already sparse
        if not sparse.issparse(X_trans):
            X_trans_sparse = sparse.csc_matrix(X_trans)
            print(" Converted X_trans to sparse CSC format for screening.")
        else:
            # Ensure it's CSC format
            X_trans_sparse = X_trans.tocsc() if not isinstance(X_trans, sparse.csc_matrix) else X_trans

        for i, rule in enumerate(self.rules):
            if not rule: continue # Skip empty rules if any slipped through

            # Check if rule items are valid column indices (0-based for slicing)
            try:
                rule_col_indices = [item - 1 for item in rule] # Convert 1-based item ID to 0-based index
                # Check if all indices are within bounds
                if not all(0 <= idx < X_trans_sparse.shape[1] for idx in rule_col_indices):
                     warnings.warn(f"Rule {i} contains invalid item indices. Skipping.")
                     continue
            except (TypeError, ValueError):
                warnings.warn(f"Rule {i} has invalid content ({rule}). Skipping.")
                continue

            # Calculate coverage efficiently using sparse matrix
            try:
                # Subselect columns for the current rule
                rule_submatrix = X_trans_sparse[:, rule_col_indices]
                # Check if all conditions in the rule are met for each row
                # Sum along axis 1: result is N_samples x 1, count of true conditions per row
                coverage_counts = rule_submatrix.sum(axis=1)
                # Rule applies if count equals the number of items in the rule
                rule_coverage_mask = np.asarray(coverage_counts == len(rule)).flatten()

                # Calculate True Positives for this rule
                tp_count = np.sum(rule_coverage_mask & (y == 1))

                # Check if support threshold is met
                if tp_count >= min_support_count:
                    # Keep this rule
                    valid_rule_indices_map[i] = new_rule_idx
                    screened_rules_temp.append(rule) # Store the rule itself
                    screened_rules_len_temp.append(len(rule)) # Store its length

                    # Store coverage data for the final RMatrix (sparse construction)
                    covered_row_indices = np.where(rule_coverage_mask)[0]
                    rows.extend(covered_row_indices)
                    cols.extend([new_rule_idx] * len(covered_row_indices))
                    data.extend([True] * len(covered_row_indices)) # Store boolean True

                    new_rule_idx += 1

            except Exception as e:
                print(f"Error processing rule {i} (Items: {rule}): {e}")
                # traceback.print_exc() # Uncomment for detailed debugging
                continue

        num_rules_after_support = new_rule_idx
        print(f" Found {num_rules_after_support} rules meeting support threshold.")

        if num_rules_after_support == 0:
            print("Warning: No rules met the support threshold.")
            self.rules = []
            self.rules_len = []
            self.supp = np.array([])
            return np.zeros((n_samples, 0), dtype=bool) # Return empty boolean matrix

        # Create the final sparse RMatrix (Rule Matrix)
        RMatrix_sparse = sparse.csc_matrix((data, (rows, cols)),
                                           shape=(n_samples, num_rules_after_support),
                                           dtype=bool)

        # Update self.rules and self.rules_len with the screened rules
        self.rules = screened_rules_temp
        self.rules_len = screened_rules_len_temp

        # --- Further screening if max_rules threshold is exceeded ---
        if num_rules_after_support > self.max_rules:
            print(f" Applying '{self.criteria}' criteria to reduce from {num_rules_after_support} to {self.max_rules} rules...")
            # Calculate metrics needed for the chosen criteria
            TP_screen = np.array(RMatrix_sparse[y == 1, :].sum(axis=0)).flatten()
            Coverage_screen = np.array(RMatrix_sparse.sum(axis=0)).flatten() # Total covered by each rule

            # Avoid division by zero
            # Precision = TP / Coverage
            precision_screen = np.divide(TP_screen, Coverage_screen,
                                         out=np.zeros_like(TP_screen, dtype=float),
                                         where=Coverage_screen != 0)

            if self.criteria == 'precision':
                scores = precision_screen
            elif self.criteria == 'information':
                # Use F1-score as a proxy for information gain (balances precision and recall)
                if n_positive == 0: # Avoid division by zero if no positive samples
                    recall_screen = np.zeros_like(TP_screen, dtype=float)
                else:
                    recall_screen = TP_screen / n_positive

                # F1 = 2 * (prec * rec) / (prec + rec)
                f1_denominator = precision_screen + recall_screen
                scores = np.divide(2 * precision_screen * recall_screen, f1_denominator,
                                   out=np.zeros_like(f1_denominator),
                                   where=f1_denominator != 0)
            else:
                warnings.warn(f"Unknown criteria '{self.criteria}', defaulting to 'precision'.")
                scores = precision_screen

            # Select indices of the top rules based on the scores
            # Use stable sort if needed, argsort is generally fine
            select_indices = np.argsort(scores)[::-1][:self.max_rules]

            # Filter rules, lengths, and the sparse RMatrix
            self.rules = [self.rules[i] for i in select_indices]
            self.rules_len = [self.rules_len[i] for i in select_indices]
            RMatrix_sparse = RMatrix_sparse[:, select_indices]
            print(f" Reduced to {len(self.rules)} rules based on '{self.criteria}'.")

        # Calculate final support counts (total coverage for remaining rules)
        self.supp = np.array(RMatrix_sparse.sum(axis=0)).flatten()

        print(f"Screening complete. Final rule count: {len(self.rules)}")

        # Return the final Rule Matrix as a dense boolean numpy array
        # Note: Dense conversion might be memory-intensive for very large datasets/rule counts
        # Consider keeping it sparse if memory becomes an issue downstream.
        return RMatrix_sparse.toarray()


    def modify_rule(self, X: pd.DataFrame, X_trans: pd.DataFrame, y: np.ndarray, rule_idx: int) -> int:
        """
        Attempts to refine numerical cutoffs within a rule.
        NOTE: The logic here is complex and based on the original script's attempt.
              It has been simplified in the Colab script provided by the user.
              This function currently implements the SIMPLIFIED version:
              It identifies numeric/categoric parts but doesn't actually modify cutoffs.
              It mainly stores the rule meaning and calculates its coverage vector Z.
        """
        if not (0 <= rule_idx < len(self.rules)):
            warnings.warn(f"modify_rule called with invalid index {rule_idx}")
            return rule_idx
        rule = self.rules[rule_idx]
        if not rule:
            self.rule_explainations[rule_idx] = ([], np.zeros(X_trans.shape[0], dtype=bool))
            return rule_idx

        # Separate numerical and categorical items based on itemNames format
        num_items = []
        cat_items = []
        item_names_in_rule = []
        valid_rule_items = True
        for item_idx in rule:
            item_name = self.itemNames.get(item_idx)
            if item_name is None:
                warnings.warn(f"Invalid item index {item_idx} found in rule {rule_idx}. Skipping item.")
                valid_rule_items = False
                continue # Skip this invalid item index

            item_names_in_rule.append(item_name)
            if '_' in item_name:
                cat_items.append(item_idx)
            elif '<' in item_name:
                num_items.append(item_idx)
            # else: could be a binary feature derived from a 2-category feature, treat as categorical?
            # For simplicity, assume items with neither '_' nor '<' are unlikely or handled implicitly.

        if not valid_rule_items and not item_names_in_rule: # Rule became empty due to invalid items
            self.rule_explainations[rule_idx] = ([], np.zeros(X_trans.shape[0], dtype=bool))
            return rule_idx

        # --- Simplified Logic from Colab Script ---
        # Calculate the final coverage vector (Z) for this rule based on X_trans
        # without attempting cutoff adjustments.
        final_rule_Z = np.ones(X_trans.shape[0], dtype=bool)
        rule_meaning_list = [] # Store the human-readable conditions

        for item_idx in rule: # Iterate through the original valid items
            item_name = self.itemNames.get(item_idx)
            if item_name and item_name in X_trans.columns:
                try:
                    final_rule_Z &= X_trans[item_name].values
                    rule_meaning_list.append(item_name)
                except KeyError:
                     # This should theoretically not happen if item_name is in X_trans.columns
                     warnings.warn(f"KeyError accessing column '{item_name}' for rule {rule_idx} during Z calculation.")
                     final_rule_Z[:] = False # Rule cannot be evaluated if column is missing
                     rule_meaning_list = [] # Invalidate meaning
                     break
                except Exception as e:
                    warnings.warn(f"Error applying condition '{item_name}' for rule {rule_idx}: {e}")
                    final_rule_Z[:] = False # Mark as unevaluable on error
                    rule_meaning_list = []
                    break
            elif item_name:
                # Column for this condition doesn't exist in X_trans (e.g., rare category)
                 warnings.warn(f"Condition '{item_name}' for rule {rule_idx} not found in X_trans columns. Rule will always be False.")
                 final_rule_Z[:] = False
                 rule_meaning_list = []
                 break
            # else: item_idx was invalid, already warned


        # Store the potentially simplified meaning and the calculated coverage vector
        self.rule_explainations[rule_idx] = (rule_meaning_list, final_rule_Z)

        return rule_idx # Return original index, as no modification happened


    def greedy_init(self, X: pd.DataFrame, X_trans: pd.DataFrame, y: np.ndarray, RMatrix: np.ndarray) -> list:
        """Provides a simple greedy initialization for the MCMC search (Optional)."""
        # Note: This implementation is simplified based on the Colab script.
        # A more sophisticated greedy approach might consider coverage of remaining positives.
        print("Starting simplified greedy initialization...")
        greedy_rules = []
        n_samples, n_rules_screened = RMatrix.shape

        if n_rules_screened == 0:
            print("Warning: RMatrix has no columns in greedy_init.")
            return []

        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            print("Warning: No positive samples in y for greedy_init.")
            return []

        precisions = []
        for i in range(n_rules_screened):
            try:
                rule_coverage_mask = RMatrix[:, i] # Already boolean
                pos_covered = np.sum(rule_coverage_mask[pos_idx])
                total_covered = np.sum(rule_coverage_mask)

                if total_covered > 0:
                    precision = pos_covered / total_covered
                    support_count = pos_covered # Use TP count as raw support
                    # Optional: Add minimum precision threshold? e.g., if precision > 0.5:
                    precisions.append({'idx': i, 'precision': precision, 'support': support_count})
            except Exception as e:
                print(f"Error evaluating rule {i} in greedy_init: {e}")

        if not precisions:
            print("No suitable rules found for greedy initialization.")
            return []

        # Sort rules: primary key precision (desc), secondary key support (desc)
        precisions.sort(key=lambda x: (x['precision'], x['support']), reverse=True)

        # Select top N rules (e.g., top 5)
        num_greedy_rules = min(5, len(precisions)) # Take up to 5 rules
        greedy_rules = [p['idx'] for p in precisions[:num_greedy_rules]]

        print(f"Greedy initialization selected {len(greedy_rules)} rules:")
        for rule_data in precisions[:num_greedy_rules]:
            print(f" Rule {rule_data['idx']}: Precision={rule_data['precision']:.4f}, Support={rule_data['support']}")

        return greedy_rules


    def normalize(self, rules_new: list) -> list:
        """Removes redundant rules (subsets/supersets) from a list of rule indices."""
        if not rules_new or len(rules_new) < 2:
            return rules_new[:] # Return copy if empty or only one rule

        try:
            # Filter out invalid indices before processing
            max_rule_index = len(self.rules) - 1
            valid_rules_indices = [idx for idx in rules_new if 0 <= idx <= max_rule_index]
            if not valid_rules_indices: return []

            # Get the actual rule item lists for valid indices
            valid_rules_items = {idx: set(self.rules[idx]) for idx in valid_rules_indices}
            rules_to_process = valid_rules_indices[:] # Work on a copy

            # Sort by length descending (longer rules first) - helps find supersets first
            rules_to_process.sort(key=lambda idx: len(valid_rules_items[idx]), reverse=True)

            final_rules = []
            for i in range(len(rules_to_process)):
                idx1 = rules_to_process[i]
                if idx1 is None: continue # Skip if already marked for removal

                is_subsumed = False
                # Check if idx1 is a subset of any rule already added to final_rules
                for kept_idx in final_rules:
                    if valid_rules_items[idx1].issubset(valid_rules_items[kept_idx]):
                         is_subsumed = True
                         break
                if is_subsumed:
                    continue # Don't add idx1

                # Check subsequent rules (idx2) to see if idx1 subsumes them
                for j in range(i + 1, len(rules_to_process)):
                    idx2 = rules_to_process[j]
                    if idx2 is None: continue

                    # If idx2 is a subset of idx1, mark idx2 for removal (None)
                    if valid_rules_items[idx2].issubset(valid_rules_items[idx1]):
                        rules_to_process[j] = None

                # If idx1 wasn't subsumed, add it to the final list
                final_rules.append(idx1)

            return final_rules

        except IndexError:
            warnings.warn("IndexError during normalization, possibly due to rule screening changes. Returning original list.")
            return rules_new[:] # Return original list on error
        except Exception as e:
            warnings.warn(f"Error during normalization: {e}. Returning original list.")
            return rules_new[:]


    def find_rules_Z(self, RMatrix: np.ndarray, rules: list) -> np.ndarray:
        """Calculates the combined coverage vector (Z) for a given set of rule indices."""
        if not rules:
            return np.zeros(RMatrix.shape[0], dtype=bool)

        Z = np.zeros(RMatrix.shape[0], dtype=bool)
        max_rule_idx = RMatrix.shape[1] - 1

        for rule_idx in rules:
            if not (0 <= rule_idx <= max_rule_idx):
                 warnings.warn(f"Invalid rule index {rule_idx} encountered in find_rules_Z. Skipping.")
                 continue

            try:
                rule_explanation = self.rule_explainations.get(rule_idx)
                if rule_explanation is None:
                    # Use coverage directly from RMatrix if not modified/explained
                    rule_coverage = RMatrix[:, rule_idx] # Assumes RMatrix is boolean
                else:
                    # Use the stored coverage vector from rule_explainations
                    rule_coverage = rule_explanation[1] # Stored Z vector

                # Ensure rule_coverage is a 1D boolean array matching RMatrix rows
                if rule_coverage.shape == (RMatrix.shape[0],):
                    Z |= rule_coverage.astype(bool) # Combine using logical OR
                else:
                     warnings.warn(f"Coverage vector shape mismatch for rule {rule_idx}. Skipping.")

            except Exception as e:
                warnings.warn(f"Error processing rule {rule_idx} in find_rules_Z: {e}")
                continue # Skip rule on error

        return Z


    def propose(self, rules_curr: list, rules_norm: list,
                nRules: int, X: pd.DataFrame, X_trans: pd.DataFrame,
                y: np.ndarray, RMatrix: np.ndarray) -> tuple[list, list]:
        """Proposes a new ruleset state (add, cut, or clean) for MCMC."""

        yhat = self.find_rules_Z(RMatrix, rules_curr)
        incorr = np.where(y != yhat)[0] # Indices of misclassified samples

        # --- Determine Move Type ---
        if len(incorr) == 0: # Perfect classification -> try cleaning
            move = ['clean']
            ex = 0 # Placeholder example index
        else:
            # Select a misclassified example
            ex = sample(list(incorr), 1)[0]
            t = random() # Random number for decision

            if y[ex] == 1: # Misclassified positive (False Negative) -> Needs more rules
                 move = ['add'] if t < 0.7 else ['cut', 'add'] # Prioritize adding
            else: # Misclassified negative (False Positive) -> Needs fewer/better rules
                 move = ['cut'] if t < 0.7 else ['cut', 'add'] # Prioritize cutting

            # Edge case overrides:
            TP, FP, TN, FN = get_confusion(yhat, y)
            current_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            # Force cut if precision is low despite covering all positives? (Heuristic)
            # if len(rules_curr) > 0 and (TP + FN == np.sum(y)) and current_precision < 0.75:
            #    move = ['cut']
            # Force add if no rules or nothing covered
            if not rules_curr or (TP + FP == 0):
                move = ['add']


        # --- Execute Move ---
        rules_new = rules_curr[:] # Start with a copy
        rules_new_norm = rules_norm[:]

        # --- Cut Move ---
        if 'cut' in move and rules_new: # Can only cut if rules exist
            try:
                candidates_to_cut = []
                # If misclassified negative (FP), prioritize cutting rules covering it
                if y[ex] == 0:
                    for rule_idx in rules_new:
                         if 0 <= rule_idx < RMatrix.shape[1] and RMatrix[ex, rule_idx]:
                             candidates_to_cut.append(rule_idx)

                # If no specific candidates or misclassified positive (FN), choose from all current rules
                if not candidates_to_cut:
                    candidates_to_cut = rules_new

                if candidates_to_cut: # Ensure there's something to cut
                    # Simple random choice for now
                    # TODO: Could implement probabilistic cut based on rule quality/redundancy
                    cut_rule = sample(candidates_to_cut, 1)[0]
                    rules_new.remove(cut_rule)
                    # Normalization happens after potential add step
            except IndexError: # Catch potential sample error if candidates_to_cut is empty
                warnings.warn("Could not select rule to cut.")
            except Exception as e:
                 warnings.warn(f"Exception during cut move: {e}")

        # --- Add Move ---
        if 'add' in move:
            try:
                available_rules_idx = list(set(range(nRules)) - set(rules_new))
                if not available_rules_idx:
                    warnings.warn("No available rules left to add.")
                else:
                    candidates_to_add = []
                    # If misclassified positive (FN), prioritize rules covering it
                    if y[ex] == 1:
                         for rule_idx in available_rules_idx:
                             if 0 <= rule_idx < RMatrix.shape[1] and RMatrix[ex, rule_idx]:
                                  candidates_to_add.append(rule_idx)

                    # If no specific candidates or misclassified negative (FP),
                    # consider adding high-precision rules.
                    if not candidates_to_add:
                         # Find high-precision rules among available ones
                         precisions_add = []
                         pos_idx_add = np.where(y == 1)[0]
                         for rule_idx in available_rules_idx:
                              if 0 <= rule_idx < RMatrix.shape[1]:
                                   rule_cov_add = RMatrix[:, rule_idx]
                                   tp_add = np.sum(rule_cov_add[pos_idx_add])
                                   total_cov_add = np.sum(rule_cov_add)
                                   if total_cov_add > 0:
                                       prec_add = tp_add / total_cov_add
                                       if prec_add > 0.6: # Add precision threshold?
                                           precisions_add.append({'idx': rule_idx, 'prec': prec_add, 'tp': tp_add})
                         if precisions_add:
                             precisions_add.sort(key=lambda x: (x['prec'], x['tp']), reverse=True)
                             # Select from top candidates
                             candidates_to_add = [p['idx'] for p in precisions_add[:max(5, len(available_rules_idx)//10)]]

                    # If still no candidates, choose randomly from available
                    if not candidates_to_add:
                        candidates_to_add = available_rules_idx

                    if candidates_to_add:
                        # Exploration vs Exploitation for adding
                        if random() < self.propose_threshold: # Explore
                             add_rule = sample(candidates_to_add, 1)[0]
                        else: # Exploit: Add rule covering most *uncovered* positives
                             yhat_current_add = self.find_rules_Z(RMatrix, rules_new) # Coverage before adding
                             uncovered_pos_idx = np.where((y == 1) & ~yhat_current_add)[0]
                             best_add_rule = -1
                             max_new_coverage = -1
                             if len(uncovered_pos_idx) > 0:
                                 for r_idx in candidates_to_add:
                                      if 0 <= r_idx < RMatrix.shape[1]:
                                           new_cov = np.sum(RMatrix[uncovered_pos_idx, r_idx])
                                           if new_cov > max_new_coverage:
                                               max_new_coverage = new_cov
                                               best_add_rule = r_idx
                             # If best found, use it, else fallback to random from candidates
                             add_rule = best_add_rule if best_add_rule != -1 else sample(candidates_to_add, 1)[0]

                        rules_new.append(add_rule)
                        # Optional: Call modify_rule here if adjustment is desired immediately
                        # if self.rule_adjust and not self.binary_input:
                        #    self.modify_rule(X, X_trans, y, add_rule)

            except IndexError: # Catch potential sample error
                 warnings.warn("Could not select rule to add.")
            except Exception as e:
                 warnings.warn(f"Exception during add move: {e}")

        # --- Normalize Final Proposed Ruleset ---
        # Normalization removes redundant rules after add/cut
        rules_new_norm = self.normalize(rules_new)

        # --- Clean Move (Normalization Only) ---
        if 'clean' in move:
            # Already normalized above if perfect classification
            rules_new_norm = self.normalize(rules_new) # Ensure normalization happens

        return rules_new, rules_new_norm


    def compute_prob(self, RMatrix: np.ndarray, y: np.ndarray, rules: list) -> tuple[list, list]:
        """Computes the components of the posterior probability for a given ruleset."""
        if not isinstance(y, np.ndarray):
             y = self.convert_y(y)

        try:
            yhat = self.find_rules_Z(RMatrix, rules)
            self.yhat = yhat # Store prediction for potential later use
            TP, FP, TN, FN = get_confusion(yhat, y)

            # Calculate Prior P(M) based on rule lengths
            prior_ChsRules = 0.0
            if self.patternSpace is not None: # Ensure patternSpace is initialized
                 # Count number of rules of each length in the current set 'rules'
                 Kn_count = np.zeros(self.maxlen + 1, dtype=int)
                 valid_lengths_count = 0
                 for rule_idx in rules:
                     if 0 <= rule_idx < len(self.rules_len):
                          length = self.rules_len[rule_idx]
                          if 1 <= length <= self.maxlen:
                               Kn_count[length] += 1
                               valid_lengths_count += 1

                 if valid_lengths_count > 0: # Only compute if there are valid rules
                     for i in range(1, self.maxlen + 1):
                          if self.patternSpace[i] > 0: # Check pattern space for this length
                               prior_ChsRules += log_betabin(Kn_count[i], self.patternSpace[i],
                                                            self.alpha_l[i], self.beta_l[i])
                 else: # If rules list is empty or only contains invalid indices
                      prior_ChsRules = self.P0 # Prior probability of empty set

            else: # Fallback if patternSpace not set (shouldn't happen in normal flow)
                 prior_ChsRules = -np.inf # Log(0)


            # Calculate Likelihood P(S|M)
            # Component 1: P(TP, FP | M) using BetaBinomial (Positives)
            # Total "trials" for positives is TP + FP (number predicted positive)
            # "Successes" is TP
            likelihood_1 = log_betabin(TP, TP + FP, self.alpha_1, self.beta_1)

            # Component 2: P(TN, FN | M) using BetaBinomial (Negatives)
            # Total "trials" for negatives is TN + FN (number predicted negative)
            # "Successes" is TN
            likelihood_2 = log_betabin(TN, TN + FN, self.alpha_2, self.beta_2)

            # Return confusion matrix and log-probability components
            return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

        except Exception as e:
            warnings.warn(f"Error in compute_prob: {e}")
            # Return default values indicating error (e.g., high FP/FN, very low probability)
            n_samples = len(y)
            n_pos = np.sum(y)
            return [0, n_samples - n_pos, 0, n_pos], [-1e9] * 3 # Large negative log-prob


    def print_rules(self, rule_indices: list):
        """Prints the rules corresponding to the given indices in a readable format."""
        print("\n--- Rule Set ---")
        if not rule_indices:
            print(" (No rules in set)")
            return

        max_rule_idx = len(self.rules) - 1
        for i, rule_idx in enumerate(rule_indices):
            if not (0 <= rule_idx <= max_rule_idx):
                print(f" Rule {i+1} (Index {rule_idx}): Invalid rule index")
                continue

            try:
                # Try getting the potentially modified explanation first
                rule_explanation = self.rule_explainations.get(rule_idx)
                if rule_explanation and rule_explanation[0]: # Use explained items if available
                    rule_items_names = rule_explanation[0]
                else: # Fallback to original items from itemNames
                    rule_items_names = [self.itemNames.get(item, f"InvalidItem({item})")
                                        for item in self.rules[rule_idx]]

                # Rewrite into human-readable format
                reformatted_conditions = self.rewrite_rules(rule_items_names)
                rule_str = ' AND '.join(reformatted_conditions)
                print(f" Rule {i+1} (Index {rule_idx}): {rule_str}")

            except Exception as e:
                print(f"Error printing rule {i+1} (Index {rule_idx}): {e}")
        print("----------------")


    def rewrite_rules(self, rules_list: list) -> list:
        """Rewrites internal rule item strings into a more human-readable format."""
        rewritten_conditions = []
        if not self.attributeNames: # Need original attribute names for context
            warnings.warn("attributeNames not set. Rule rewriting might be inaccurate.")
            return rules_list # Return as is if attributes aren't known

        for rule_item_str in rules_list:
            if not isinstance(rule_item_str, str):
                rewritten_conditions.append(f"[InvalidType: {rule_item_str}]")
                continue

            try:
                # --- Handle Numerical Conditions (e.g., "Age<25.5", "50.0<Income") ---
                if '<' in rule_item_str:
                    parts = rule_item_str.split('<', 1)
                    if len(parts) == 2:
                        part1, part2 = parts[0], parts[1]
                        # Try converting parts to float to identify numbers vs feature names
                        try: val1 = float(part1); is_part1_numeric = True
                        except ValueError: is_part1_numeric = False
                        try: val2 = float(part2); is_part2_numeric = True
                        except ValueError: is_part2_numeric = False

                        # Case 1: Feature < Value (e.g., "Age<25.5")
                        # Check if part1 is a known original attribute name
                        if part1 in self.attributeNames and is_part2_numeric:
                             rewritten_conditions.append(f"{part1} < {val2:.3g}") # Use .3g for clean format
                             continue
                        # Case 2: Value < Feature (e.g., "50.0<Income" -> means Income >= 50.0)
                        # Check if part2 is a known original attribute name
                        elif part2 in self.attributeNames and is_part1_numeric:
                             rewritten_conditions.append(f"{part2} >= {val1:.3g}")
                             continue
                        # Case 3: Ambiguous or complex interval (e.g. from internal transform names)
                        # Try to parse based on original attribute names being present
                        else:
                            found_orig_attr = False
                            for attr_name in self.attributeNames:
                                if rule_item_str.startswith(f"{attr_name}<"): # Feature < Value
                                     try:
                                         value_str = rule_item_str.split('<', 1)[1]
                                         value = float(value_str)
                                         rewritten_conditions.append(f"{attr_name} < {value:.3g}")
                                         found_orig_attr = True
                                         break
                                     except (ValueError, IndexError): pass
                                elif rule_item_str.endswith(f"<{attr_name}"): # Value < Feature
                                     try:
                                         value_str = rule_item_str.split('<', 1)[0]
                                         value = float(value_str)
                                         rewritten_conditions.append(f"{attr_name} >= {value:.3g}")
                                         found_orig_attr = True
                                         break
                                     except (ValueError, IndexError): pass
                            if found_orig_attr: continue

                        # If parsing failed, keep original string but maybe mark it
                        rewritten_conditions.append(f"[{rule_item_str}]") # Mark as potentially unparsed

                    else: # Incorrect format
                         rewritten_conditions.append(f"[InvalidNumFmt: {rule_item_str}]")

                # --- Handle Categorical Conditions (e.g., "Color_Red", "Color_Red_neg") ---
                elif '_' in rule_item_str:
                    # Find the longest original attribute name that is a prefix
                    potential_feat = ""
                    best_match_len = -1
                    for attr_name in self.attributeNames:
                         prefix = f"{attr_name}_"
                         if rule_item_str.startswith(prefix):
                             if len(attr_name) > best_match_len:
                                  potential_feat = attr_name
                                  best_match_len = len(attr_name)

                    if potential_feat: # Found a likely original feature name
                        value_part = rule_item_str[len(potential_feat) + 1:] # Get the part after "feat_"
                        if value_part.endswith('_neg'):
                            val = value_part[:-4] # Remove '_neg' suffix
                            rewritten_conditions.append(f"{potential_feat} is not '{val}'")
                        else:
                            rewritten_conditions.append(f"{potential_feat} is '{value_part}'")
                    else: # Couldn't identify original feature based on known attributes
                        rewritten_conditions.append(f"[UnknownCat: {rule_item_str}]")

                # --- Handle items that seem like direct binary features ---
                # (e.g., result of transform for a 2-category feature like "Gender_Male")
                else:
                     # Check if the item name itself matches a transformed column name
                     # This implies the condition is "FeatureName is True"
                     # We can simplify this by just stating the condition name
                     rewritten_conditions.append(rule_item_str)

            except Exception as e:
                 warnings.warn(f"Error rewriting rule item '{rule_item_str}': {e}")
                 rewritten_conditions.append(f"[RewriteError: {rule_item_str}]")

        return rewritten_conditions


    def Bayesian_patternbased(self, X: pd.DataFrame, X_trans: pd.DataFrame,
                              y: np.ndarray, RMatrix: np.ndarray, init_rules: list):
        """Performs the MCMC search to find the optimal ruleset."""
        if RMatrix.shape[1] == 0:
            print("Warning: RMatrix is empty, cannot perform MAP search.")
            self.predicted_rules = []
            return defaultdict(list) # Return empty map history

        nRules = RMatrix.shape[1] # Number of candidate rules after screening

        # --- Initialization ---
        # Bounds for potential pruning (Paper Section 3.3) - currently not strictly enforced in proposal
        # self.Asize = [[min(self.patternSpace[l]/2,
        #                    0.5*(self.patternSpace[l]+self.beta_l[l]-self.alpha_l[l]))
        #                    for l in range(self.maxlen+1) if self.patternSpace[l]>0]]
        # self.C = [1] # Support threshold bound

        self.maps = defaultdict(list) # Store results for chain 0
        T0 = 1000.0 # Initial temperature for simulated annealing
        early_stop_patience = 5000 # Iterations without improvement to stop
        no_improvement_count = 0

        # Start with initial ruleset (greedy or empty)
        if self.greedy_initilization and init_rules:
             # Ensure init_rules contains valid indices for the current RMatrix
             rules_curr = [r for r in init_rules if 0 <= r < nRules]
             if len(rules_curr) != len(init_rules):
                  warnings.warn("Some initial rules from greedy_init were invalid after screening.")
             print(f"Starting MAP search with {len(rules_curr)} initial rules from greedy search.")
        else:
             rules_curr = []
             print("Starting MAP search with empty ruleset.")

        # Normalize the initial ruleset
        rules_curr_norm = self.normalize(rules_curr)

        # Evaluate initial state
        cfmatrix_curr, prob_curr = self.compute_prob(RMatrix, y, rules_curr)
        pt_curr = sum(prob_curr) # Current log posterior score
        print(f"Initial ruleset score: {pt_curr:.4f}, "
              f"Accuracy: {(cfmatrix_curr[0]+cfmatrix_curr[2])/len(y):.4f}, "
              f"Rules: {len(rules_curr)}")

        # Store best state found so far
        best_score = pt_curr
        best_rules = rules_curr[:] # Use copy
        # Store initial state in maps history
        self.maps[0].append([-1, prob_curr, best_rules,
                            [self.rules[i] for i in best_rules], # Store actual rule items
                            cfmatrix_curr])
        self.predicted_rules = best_rules # Initialize final predicted rules

        # --- MCMC Loop ---
        print(f"\nStarting MCMC search for {self.max_iter} iterations...")
        for ith_iter in range(self.max_iter):
            # 1. Propose a new state (rules_new, rules_new_norm)
            rules_new, rules_new_norm = self.propose(rules_curr, rules_curr_norm, # Pass copies
                                                     nRules, X, X_trans, y, RMatrix)

            # 2. Compute probability of the new state
            cfmatrix_new, prob_new = self.compute_prob(RMatrix, y, rules_new)
            pt_new = sum(prob_new)

            # 3. Calculate Acceptance Probability (Metropolis-Hastings)
            T = T0 * (1 - (ith_iter + 1) / self.max_iter) # Annealing schedule
            T = max(T, 1e-9) # Ensure temperature doesn't drop below zero

            delta_E = pt_new - pt_curr # Change in log posterior
            accept_prob = 1.0 if delta_E > 0 else np.exp(delta_E / T)

            # Log progress periodically
            if (ith_iter + 1) % 1000 == 0:
                 accuracy = (cfmatrix_new[0]+cfmatrix_new[2]) / len(y) if len(y) > 0 else 0
                 print(f"Iter {ith_iter+1:>{len(str(self.max_iter))}}: "
                       f"Score={pt_new:.3f} (Best={best_score:.3f}), "
                       f"Acc={accuracy:.4f}, Rules={len(rules_new)}, "
                       f"Temp={T:.3f}, AcceptProb={accept_prob:.4f}")

            # 4. Check if new best solution found
            if pt_new > best_score:
                 print(f"  ** New best at iter = {ith_iter+1} -> Score: {pt_new:.4f} **")
                 best_score = pt_new
                 best_rules = rules_new[:] # Store copy
                 no_improvement_count = 0 # Reset patience counter

                 # Store this improved state in maps
                 # Use the normalized version for storage? Or the direct proposal? Let's store direct proposal.
                 self.maps[0].append([ith_iter, prob_new, best_rules,
                                     [self.rules[i] for i in best_rules if 0 <= i < len(self.rules)], # Store items safely
                                     cfmatrix_new])
                 self.predicted_rules = best_rules # Update main predicted rules

                 # Optionally print the new best ruleset
                 # self.print_rules(self.normalize(best_rules)) # Print normalized version

                 # --- Optional: Update Bounds (Asize, C) ---
                 # This part implements the bounds update from the paper, which could
                 # potentially prune the search space but adds complexity.
                 # It's currently disabled/simplified in the propose step.
                 # try:
                 #    # Update Asize bound (simplified)
                 # except Exception as e_bound:
                 #    warnings.warn(f"Error updating bounds Asize/C: {e_bound}")

            else:
                 no_improvement_count += 1

            # 5. Accept or Reject the move
            if random() <= accept_prob:
                # Accept the new state
                rules_curr = rules_new[:]
                rules_curr_norm = rules_new_norm[:]
                pt_curr = pt_new
                # Note: cfmatrix_curr isn't strictly needed for the loop logic after acceptance prob calculation

            # Else: Reject the move, rules_curr/norm/pt_curr remain unchanged

            # 6. Check for Early Stopping
            if no_improvement_count >= early_stop_patience:
                print(f"\nNo improvement in best score for {no_improvement_count} iterations. Early stopping at iter {ith_iter+1}.")
                break

        # --- Post-Loop ---
        # Ensure self.predicted_rules holds the best ruleset found during the entire search
        if self.maps[0]: # Check if any states were recorded
             # Find the entry in maps with the highest score
             best_map_entry = max(self.maps[0], key=lambda x: sum(x[1]) if len(x) > 1 and isinstance(x[1], list) else -np.inf)
             # Check if map entry is valid and compare score
             if len(best_map_entry) > 2 and isinstance(best_map_entry[1], list):
                  best_recorded_score = sum(best_map_entry[1])
                  if best_recorded_score > best_score:
                       print(f"Reverting to best recorded map score: {best_recorded_score:.4f}")
                       self.predicted_rules = best_map_entry[2] # Use rules from the best map entry
                  # else: best_score from live tracking was already better or equal

        print(f"\nMCMC search complete. Final best score: {best_score:.4f}")

        # Clean the final predicted rules (e.g., remove rules with zero coverage after all adjustments)
        # Note: clean_rules implementation needs careful review
        self.predicted_rules = self.clean_rules(RMatrix, self.predicted_rules)
        print(f"Final predicted rules after cleaning: {len(self.predicted_rules)}")

        return self.maps[0] # Return the history for chain 0


    def ensure_minimum_rules(self, RMatrix: np.ndarray, y: np.ndarray, min_rules: int = 3):
        """If the final ruleset is too small, attempts to add high-quality rules."""
        print(f"\nChecking minimum rules (Target: {min_rules})...")
        y = self.convert_y(y)

        if not hasattr(self, 'predicted_rules'):
            self.predicted_rules = [] # Should not happen if fit was called

        if len(self.predicted_rules) >= min_rules:
            print(f" Found {len(self.predicted_rules)} rules, which meets the minimum. Skipping additions.")
            return

        print(f" Current ruleset size ({len(self.predicted_rules)}) is less than minimum ({min_rules}). Trying to add rules...")

        # --- Identify Candidate Rules to Add ---
        # Start with all screened rules that are not currently predicted
        candidate_pool_idx = list(set(range(len(self.rules))) - set(self.predicted_rules))
        if not candidate_pool_idx:
            print(" No more candidate rules available to add.")
            return

        # Evaluate candidates based on precision and support (TP count)
        candidate_metrics = []
        n_positive = np.sum(y)
        for rule_idx in candidate_pool_idx:
            try:
                 if 0 <= rule_idx < RMatrix.shape[1]:
                     rule_coverage = RMatrix[:, rule_idx]
                     tp_rule = np.sum(rule_coverage & y)
                     fp_rule = np.sum(rule_coverage & ~y)
                     total_coverage = tp_rule + fp_rule

                     if total_coverage > 0: # Consider rules with non-zero coverage
                          precision_rule = tp_rule / total_coverage
                          # Add thresholds: e.g., min precision, min TP count?
                          # Use support percentage relative to positive class?
                          min_support_pct_ensure = 1.0 # Require at least 1% support of positive class
                          min_support_cnt_ensure = math.ceil((min_support_pct_ensure / 100.0) * n_positive) if n_positive > 0 else 1

                          if precision_rule > 0.55 and tp_rule >= min_support_cnt_ensure: # Example thresholds
                               candidate_metrics.append({'idx': rule_idx, 'prec': precision_rule, 'tp': tp_rule})
            except Exception as e:
                 warnings.warn(f"Error evaluating candidate rule {rule_idx} in ensure_minimum_rules: {e}")

        if not candidate_metrics:
             print(" No suitable candidate rules found to add.")
             return

        # Sort candidates: best precision first, then highest TP count
        candidate_metrics.sort(key=lambda x: (x['prec'], x['tp']), reverse=True)
        print(f" Found {len(candidate_metrics)} candidate rules to potentially add.")

        # --- Iteratively Add Rules ---
        current_rules = self.predicted_rules[:] # Work on a copy
        added_count = 0

        # Calculate initial F1 score
        yhat_current = self.find_rules_Z(RMatrix, current_rules)
        tp_curr, fp_curr, tn_curr, fn_curr = get_confusion(yhat_current, y)
        prec_curr = tp_curr / (tp_curr + fp_curr) if (tp_curr + fp_curr) > 0 else 0
        rec_curr = tp_curr / (tp_curr + fn_curr) if (tp_curr + fn_curr) > 0 else 0
        best_f1 = 2 * prec_curr * rec_curr / (prec_curr + rec_curr) if (prec_curr + rec_curr) > 0 else 0

        for candidate in candidate_metrics:
             if len(current_rules) >= min_rules:
                 break # Stop if minimum count reached

             rule_idx_to_add = candidate['idx']
             # Avoid adding duplicates if somehow present
             if rule_idx_to_add in current_rules: continue

             # Evaluate F1 score *if* this rule is added
             temp_rules = current_rules + [rule_idx_to_add]
             yhat_temp = self.find_rules_Z(RMatrix, temp_rules)
             tp_temp, fp_temp, tn_temp, fn_temp = get_confusion(yhat_temp, y)
             prec_temp = tp_temp / (tp_temp + fp_temp) if (tp_temp + fp_temp) > 0 else 0
             rec_temp = tp_temp / (tp_temp + fn_temp) if (tp_temp + fn_temp) > 0 else 0
             f1_temp = 2 * prec_temp * rec_temp / (prec_temp + rec_temp) if (prec_temp + rec_temp) > 0 else 0

             # Add rule if it doesn't significantly hurt F1 (or improves it)
             # Allow a small tolerance (e.g., 0.005 drop)
             if f1_temp >= best_f1 - 0.005:
                 print(f" Adding rule {rule_idx_to_add} (Prec: {candidate['prec']:.3f}). New F1: {f1_temp:.4f} (vs {best_f1:.4f})")
                 current_rules.append(rule_idx_to_add)
                 best_f1 = max(best_f1, f1_temp) # Update best F1 if improved
                 added_count += 1
             # else:
             #    print(f" Skipping rule {rule_idx_to_add}, F1 decreased: {f1_temp:.4f} < {best_f1:.4f}")

        print(f" Added {added_count} rules during ensure_minimum_rules.")
        self.predicted_rules = current_rules # Update the final ruleset


    def clean_rules(self, RMatrix: np.ndarray, rules: list) -> list:
        """Removes rules with zero coverage from the final list."""
        if not rules: return []

        cleaned = []
        rules_with_coverage = []
        max_rule_idx = RMatrix.shape[1] - 1

        for rule_idx in rules:
             if not (0 <= rule_idx <= max_rule_idx):
                  warnings.warn(f"Invalid rule index {rule_idx} found during cleaning. Skipping.")
                  continue

             try:
                 # Check coverage using RMatrix (which reflects screened rules)
                 coverage = np.sum(RMatrix[:, rule_idx])
                 if coverage > 0:
                      # Optional: Add minimum coverage threshold?
                      # min_coverage_threshold = 5 # Example
                      # if coverage >= min_coverage_threshold:
                      #      rules_with_coverage.append({'idx': rule_idx, 'coverage': coverage})
                      rules_with_coverage.append({'idx': rule_idx, 'coverage': coverage})
                 # else: Rule has zero coverage, implicitly dropped
             except IndexError:
                  warnings.warn(f"IndexError accessing RMatrix for rule {rule_idx} during cleaning.")
             except Exception as e:
                 warnings.warn(f"Error checking coverage for rule {rule_idx} during cleaning: {e}")

        if not rules_with_coverage:
             print("Warning: No rules with coverage found after cleaning.")
             return []

        # Return indices of rules with non-zero coverage
        cleaned = [r['idx'] for r in rules_with_coverage]

        # Optional: If all rules were below a threshold but had some coverage, keep the best?
        # Example: (Implement if using min_coverage_threshold logic above)
        # if not cleaned and rules_with_coverage: # If threshold applied and removed all
        #     rules_with_coverage.sort(key=lambda x: x['coverage'], reverse=True)
        #     best_rule_idx = rules_with_coverage[0]['idx']
        #     print(f"All rules below coverage threshold, keeping best: {best_rule_idx}")
        #     cleaned = [best_rule_idx]

        if len(cleaned) < len(rules):
            print(f" Cleaned ruleset: Removed {len(rules) - len(cleaned)} rules with zero coverage.")

        return cleaned


    def fit(self, X: pd.DataFrame, y):
        """Fits the Bayesian Rule Set model to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (array-like): Training target variable (binary).

        Returns:
            self: The fitted model instance.
        """
        start_fit_time = time.time()
        print("="*50)
        print("Starting BayesianRuleSet Fit Process")
        print("="*50)

        # --- 1. Input Validation and Conversion ---
        if not isinstance(X, pd.DataFrame):
             raise TypeError("Input X must be a pandas DataFrame.")
        y_train = self.convert_y(y) # Ensure y is binary numpy array
        print(f"Fitting model on {len(y_train)} samples ({np.sum(y_train)} positive / {len(y_train)-np.sum(y_train)} negative)")

        # --- 2. Data Transformation (if not binary_input) ---
        print("\nStep 1: Transforming data...")
        start_transform = time.time()
        if self.binary_input:
            print(" binary_input=True, skipping transformation.")
            X_trans = X.copy() # Use a copy
            # Ensure boolean type if specified as binary
            if not np.issubdtype(X_trans.values.dtype, np.bool_):
                 try:
                     X_trans = X_trans.astype(bool)
                 except ValueError:
                      raise ValueError("binary_input=True but data could not be converted to boolean.")
        else:
            X_trans = self.transform(X, neg=self.neg)
        end_transform = time.time()
        print(f"Data transformation took {end_transform - start_transform:.2f}s. Shape: {X_trans.shape}")

        if X_trans.shape[1] == 0:
             warnings.warn("Error: No features remained after transformation. Cannot proceed.")
             self.predicted_rules = []
             return self # Return unfit model

        # --- 3. Parameter Setup & Precomputation ---
        print("\nStep 2: Setting parameters and precomputing...")
        self.set_parameters(X_trans)
        self.precompute(y_train)
        if not self.binary_input:
            self.compute_cutoffs(X, y_train) # Compute original cutoffs for potential adjustment/interpretation

        # --- 4. Rule Generation ---
        print("\nStep 3: Generating candidate rules...")
        start_generate = time.time()
        self.generate_rules(X_trans, y_train)
        end_generate = time.time()
        print(f"Rule generation took {end_generate - start_generate:.2f}s.")

        if not self.rules:
            warnings.warn("Error: No rules were generated. Cannot proceed.")
            self.predicted_rules = []
            return self # Return unfit model

        # --- 5. Rule Screening ---
        print("\nStep 4: Screening rules...")
        start_screen = time.time()
        RMatrix = self.screen_rules(X_trans, y_train) # Returns dense boolean array
        end_screen = time.time()
        print(f"Rule screening took {end_screen - start_screen:.2f}s.")

        if RMatrix.shape[1] == 0:
            warnings.warn("Error: No rules survived screening. Cannot proceed.")
            self.predicted_rules = []
            return self # Return unfit model

        # --- 6. MCMC MAP Search ---
        print("\nStep 5: Performing MCMC MAP search...")
        self.rule_explainations = dict() # Reset explanations before MCMC
        init_mcmc_rules = []
        if self.greedy_initilization:
            start_greedy = time.time()
            # Pass original X for potential use in modify_rule during greedy steps (though simplified now)
            init_mcmc_rules = self.greedy_init(X, X_trans, y_train, RMatrix)
            print(f"Greedy initialization took {time.time() - start_greedy:.2f}s.")

        start_map = time.time()
        # Pass original X for potential use in modify_rule if called during MCMC proposals
        self.Bayesian_patternbased(X, X_trans, y_train, RMatrix, init_mcmc_rules)
        end_map = time.time()
        print(f"MAP search took {end_map - start_map:.2f}s.")

        # --- 7. Post-processing (Ensure Minimum Rules) ---
        # Apply this step cautiously, only if MAP search resulted in a very sparse model
        # or if performance metrics (calculated internally if needed) are very low.
        perform_ensure_step = False
        if len(self.predicted_rules) < 3: # Arbitrary minimum threshold
             perform_ensure_step = True
        # Optional: Calculate F1 on training data to check performance
        # yhat_final_map = self.find_rules_Z(RMatrix, self.predicted_rules)
        # tp_map, fp_map, tn_map, fn_map = get_confusion(yhat_final_map, y_train)
        # f1_map = 2*tp_map / (2*tp_map + fp_map + fn_map) if (2*tp_map + fp_map + fn_map) > 0 else 0
        # if f1_map < 0.6: # Arbitrary performance threshold
        #     perform_ensure_step = True

        if perform_ensure_step:
             print("\nStep 6: Ensuring minimum number of rules...")
             start_ensure = time.time()
             self.ensure_minimum_rules(RMatrix, y_train, min_rules=3)
             print(f"Ensure minimum rules took {time.time() - start_ensure:.2f}s.")
        else:
             print("\nStep 6: Skipping ensure_minimum_rules step.")

        # --- Fit Complete ---
        end_fit_time = time.time()
        print("\n" + "="*50)
        print(f"Fit process complete in {end_fit_time - start_fit_time:.2f} seconds.")
        print(f"Final model has {len(self.predicted_rules)} rules.")
        print("="*50)

        # Final check: Print the final selected rules
        self.print_rules(self.predicted_rules)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for new data using the fitted ruleset.

        Args:
            X (pd.DataFrame): Dataframe with features for prediction.

        Returns:
            np.ndarray: Predicted binary class labels (0 or 1).
        """
        fit_called = hasattr(self, 'itemNames') and self.itemNames is not None
        if not fit_called or not self.predicted_rules:
             warnings.warn("Model has not been fitted or has no rules. Predicting 0 for all instances.")
             return np.zeros(X.shape[0], dtype=int)

        print(f"Predicting with {len(self.predicted_rules)} rules...")

        # --- Transform Input Data ---
        if self.binary_input:
            X_trans_pred = X.copy()
             # Ensure boolean type
            if not np.issubdtype(X_trans_pred.values.dtype, np.bool_):
                 try: X_trans_pred = X_trans_pred.astype(bool)
                 except ValueError: raise ValueError("Predict input is not boolean despite binary_input=True.")
        else:
            print(" Transforming prediction data...")
            # Use the same transformation parameters (neg, level stored in self)
            X_trans_pred = self.transform(X, neg=self.neg)

        # --- Align Columns ---
        # Ensure prediction data has all columns expected by the rules (based on itemNames)
        # Columns present during training (keys are 1-based index, values are names)
        expected_col_names = list(self.itemNames.values())
        missing_cols = [col for col in expected_col_names if col not in X_trans_pred.columns]
        present_cols = [col for col in expected_col_names if col in X_trans_pred.columns]

        if missing_cols:
            warnings.warn(f"Prediction data missing {len(missing_cols)} columns expected by rules: {missing_cols[:5]}..."
                          " Rules using these columns will evaluate to False.")
            # Add missing columns and fill with False
            for col in missing_cols:
                X_trans_pred[col] = False

        # Create RMatrix for prediction
        RMatrix_pred = np.zeros((X_trans_pred.shape[0], len(self.rules)), dtype=bool)
        
        # Fill RMatrix based on rules and columns
        for rule_idx, rule in enumerate(self.rules):
            try:
                # Get rule explanation if available or use original rule
                rule_explanation = self.rule_explainations.get(rule_idx)
                if rule_explanation and rule_explanation[0]:
                    # Use stored meaning (column names)
                    rule_cols = rule_explanation[0]
                else:
                    # Use the original rule by converting item indices to column names
                    rule_cols = [self.itemNames.get(item, "") for item in rule]
                    # Remove any invalid items (empty strings)
                    rule_cols = [col for col in rule_cols if col]
                
                if not rule_cols:
                    continue  # Skip empty rules
                
                # Initialize rule coverage vector with all True
                rule_coverage = np.ones(X_trans_pred.shape[0], dtype=bool)
                
                # Apply each condition in the rule
                for col in rule_cols:
                    if col in X_trans_pred.columns:
                        rule_coverage &= X_trans_pred[col].values
                    else:
                        # Column missing means condition can't be true
                        rule_coverage[:] = False
                        break
                
                # Store coverage in RMatrix
                RMatrix_pred[:, rule_idx] = rule_coverage
                
            except Exception as e:
                warnings.warn(f"Error applying rule {rule_idx} during prediction: {e}")
                # Rule couldn't be applied, leave as False
        
        # Apply the final ruleset (predicted_rules) to get predictions
        yhat = self.find_rules_Z(RMatrix_pred, self.predicted_rules)
        
        # Convert boolean predictions to 0/1 integers
        predictions = yhat.astype(int)
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for input data X.
        
        This is a simplified implementation that returns binary probabilities
        (either 0 or 1) based on rule coverage rather than true calibrated probabilities.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Array of shape (n_samples, 2) with probabilities for [negative, positive] class
        """
        y_pred = self.predict(X)
        
        # Create probability array: [[1-p, p], [1-p, p], ...] 
        # where p is either 0 or 1 based on the prediction
        proba = np.zeros((len(y_pred), 2))
        proba[:, 1] = y_pred  # Probability of positive class
        proba[:, 0] = 1 - y_pred  # Probability of negative class
        
        return proba

    def score(self, X: pd.DataFrame, y) -> float:
        """Returns the accuracy score of the model on the given test data.
        
        Args:
            X (pd.DataFrame): Test features
            y (array-like): True labels
            
        Returns:
            float: Accuracy score (proportion of correctly classified instances)
        """
        y_true = self.convert_y(y)
        y_pred = self.predict(X)
        
        # Calculate accuracy
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        
        return correct / total if total > 0 else 0.0

    def get_rule_importances(self) -> list:
        """Calculate importance scores for the rules in the final ruleset.
        
        Returns:
            list: List of dictionaries containing rule indices and importance scores
        """
        if not hasattr(self, 'maps') or not self.maps:
            return []
        
        # Try to get the best model state from maps
        try:
            best_entry = max(self.maps[0], key=lambda x: sum(x[1]) if len(x) > 1 else -np.inf)
            
            if len(best_entry) >= 5:
                # Extract confusion matrix from the best state
                cfmatrix = best_entry[4]
                TP, FP, TN, FN = cfmatrix
                
                # Calculate total accuracy
                total_acc = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
                
                # For each rule, calculate its contribution
                rule_importances = []
                for idx, rule_idx in enumerate(self.predicted_rules):
                    # Try evaluating model without this rule
                    temp_rules = self.predicted_rules.copy()
                    temp_rules.remove(rule_idx)
                    
                    # If no rules left, assume minimum accuracy
                    if not temp_rules:
                        importance = total_acc  # Maximum importance if removing rule makes model empty
                    else:
                        # Calculate accuracy without this rule (would require re-predicting)
                        # This is an estimate, as accurate calculation would require the fitted data
                        # For simplicity, we use rule length as a proxy for importance
                        rule_len = self.rules_len[rule_idx] if rule_idx < len(self.rules_len) else 0
                        normalized_len = rule_len / self.maxlen if self.maxlen > 0 else 0
                        importance = 0.3 * normalized_len + 0.7 * (1 / len(self.predicted_rules))
                    
                    rule_importances.append({
                        'rule_idx': rule_idx,
                        'importance': importance,
                        'rule_len': self.rules_len[rule_idx] if rule_idx < len(self.rules_len) else 0
                    })
                
                # Sort by importance
                rule_importances.sort(key=lambda x: x['importance'], reverse=True)
                return rule_importances
                
        except Exception as e:
            warnings.warn(f"Error calculating rule importances: {e}")
        
        # Fallback to simple length-based importance if maps analysis fails
        return [{'rule_idx': idx, 'importance': len(self.rules[idx]) / self.maxlen if idx < len(self.rules) else 0}
                for idx in self.predicted_rules]

    def export_rules(self, output_format='text') -> str:
        """Export the rules in the specified format.
        
        Args:
            output_format (str): Format to export rules ('text' or 'json')
            
        Returns:
            str: Exported rules as string
        """
        if not self.predicted_rules:
            return "No rules found in model."
        
        if output_format == 'json':
            import json
            rules_export = []
            
            for i, rule_idx in enumerate(self.predicted_rules):
                if not (0 <= rule_idx < len(self.rules)):
                    continue
                
                # Get rule explanation if available
                rule_explanation = self.rule_explainations.get(rule_idx)
                if rule_explanation and rule_explanation[0]:
                    conditions = rule_explanation[0]
                else:
                    conditions = [self.itemNames.get(item, f"Unknown({item})")
                                 for item in self.rules[rule_idx]]
                
                # Rewrite into human-readable format
                readable_conditions = self.rewrite_rules(conditions)
                
                rules_export.append({
                    "rule_id": i+1,
                    "rule_index": rule_idx,
                    "conditions": readable_conditions,
                    "raw_conditions": conditions,
                    "length": len(self.rules[rule_idx]) if rule_idx < len(self.rules) else 0
                })
            
            return json.dumps(rules_export, indent=2)
        else:
            # Default to text format
            output = ["=== Bayesian Rule Set ==="]
            
            for i, rule_idx in enumerate(self.predicted_rules):
                if not (0 <= rule_idx < len(self.rules)):
                    output.append(f"Rule {i+1}: Invalid rule index {rule_idx}")
                    continue
                
                # Get rule explanation if available
                rule_explanation = self.rule_explainations.get(rule_idx)
                if rule_explanation and rule_explanation[0]:
                    conditions = rule_explanation[0] 
                else:
                    conditions = [self.itemNames.get(item, f"Unknown({item})")
                                 for item in self.rules[rule_idx]]
                
                # Rewrite into human-readable format
                readable_conditions = self.rewrite_rules(conditions)
                rule_str = ' AND '.join(readable_conditions)
                output.append(f"Rule {i+1}: {rule_str}")
            
            output.append("=====================")
            return "\n".join(output)

# Make the class available at the package level
__all__ = ['BayesianRuleSet']