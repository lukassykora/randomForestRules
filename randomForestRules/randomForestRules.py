import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List


class RandomForestRules:
    """
    Generate random forest and extract classification rules from trees.

    - Underscore is not allowed in column names.
    - Target must be numerical and binary.
    ...

    Attributes
    ----------
    data : pd.DataFrame
        Raw input data.
    target_values : list
        Possible target values
    count_rows: int
        Number of rows in data
    data_dummies : pd.DataFrame
        Categorical variables converted into dummy/indicator variables.
    antecedent : List[str]
        List of antecedent columns names.
    consequent : str
        Consequent column name.
    possible_values_dict : dict
        Dictionary of possible categories per each column,
    rules_frame : pd.DataFrame
        Discovered classification rules

    Methods
    -------
    read_csv(self, file: str, **kwargs)
        Import data from a CSV file.
    load_pandas(self, data_frame: pd.DataFrame)
        Import data from Pandas data frame.
    fit(self, antecedent: List[str], consequent: str, supp: float, conf: float, n_estimators: int=30, **kwargs)
        Train the model.
    get_frame(self)
        Get classification rules data frame.
    """

    def __init__(self):
        """
        Initialise.
        """
        self.data = pd.DataFrame()
        self.target_values = []
        self.count_rows = 0
        self.data_dummies = pd.DataFrame()
        self.antecedent = []
        self.consequent = ""
        self.possible_values_dict = {}
        self.rules_frame = pd.DataFrame()

    def read_csv(self, file: str, **kwargs):
        """Imports data from a CSV file.

        It uses the same optional parameters as read_csv from Pandas.

        Parameters
        ----------
        file : str
            A path to a file.
        **kwargs :
            Arbitrary keyword arguments (the same as in Pandas).
        """
        self.data = pd.read_csv(file, **kwargs)

    def load_pandas(self, data_frame: pd.DataFrame):
        """Loads a data frame.

        It must be the Pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            Pandas data frame.
        """
        self.data = data_frame

    def _prepare_data(self, antecedent: List[str], consequent: str):
        """Data preparation

        Columns are reduced just to antecedent and consequent.
        Missing values are replaced with string 'nan'.
        Possible values for each column are saved to possible_values_dict.

        Parameters
        ----------
        antecedent : List[str]
            List of antecedent columns names.
        consequent : str
            Name of consequent column.
        """
        reduced = antecedent + [consequent]
        self.data = self.data[reduced]
        self.target_values = sorted(self.data[consequent].unique())
        self.count_rows = self.data.shape[0]
        self.data = self.data.fillna("nan")
        self.antecedent = antecedent
        self.consequent = consequent
        for col in self.antecedent:
            self.possible_values_dict[col] = self.data[col].unique().tolist()

    def _set_dummies(self):
        """Get dummies and save them to new data frame.
        """
        data_reduced = self.data[self.antecedent]
        self.data_dummies = pd.get_dummies(data_reduced, columns=self.antecedent)

    def _get_random_forest(self, n_estimators: int=30, **kwargs):
        """Get random forest by Scikit-Learn RandomForestClassifier.

        Parameters
        ----------
        n_estimators : int=30
            Number of generated trees.
        **kwargs :
            Arbitrary keyword arguments for RandomForestRegressor.

        Returns
        -------
        RandomForestRegressor
            Trained random forest.
        """
        rf = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        train_x = self.data_dummies
        train_y = self.data[self.consequent].values.ravel()
        rf.fit(train_x, train_y)
        return rf

    def _get_trees(self, rf: RandomForestClassifier) -> dict:
        """Get trees from RandomForestClassifier.

        Parameters
        ----------
        rf : RandomForestClassifier
            Trained random forest.

        Returns
        -------
        dict
            Dictionary of trees.
        """
        trees = {}
        for tree_idx, est in enumerate(rf.estimators_):
            trees[tree_idx] = {}
            tree = est.tree_
            iterator = enumerate(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
            for node_idx, data in iterator:
                left, right, feature, th, value = data
                # left: index of left child (if any)
                # right: index of right child (if any)
                # feature: index of the feature to check
                # th: the threshold to compare against
                # value: values associated with classes
                trees[tree_idx][node_idx] = [feature, th, left, right]
                # for classifier, value is 0 except the index of the class to return
                # class_idx = np.argmax(value[0])
                # Tree representation
                if left == -1 and right == -1:
                    values_leaf = value[0].tolist()
                    target_index = values_leaf.index(max(values_leaf))
                    target = self.target_values[target_index]
                    trees[tree_idx][node_idx] = [feature, th, left, right, target]
                else:
                    trees[tree_idx][node_idx] = [feature, th, left, right, None]
        return trees

    def _get_tree_rules(self, trees: dict) -> list:
        """Get classification rules from tree representations.

        Parameters
        ----------
        trees : dict
            Dictionary of trees.

        Returns
        -------
        list
            List of classification rules.
        """
        tree_rules = []
        cols = self.data_dummies.columns
        for tid, tree in trees.items():
            open_rules = {0: [0, []]}
            rules = []
            max_rule = 0
            while len(open_rules) > 0:
                loop_rules = open_rules.copy()
                for key, value in loop_rules.items():
                    feature, th, left, right, target = tree[value[0]]
                    if target is None:
                        # Right
                        max_rule += 1
                        open_rules[max_rule] = [None, []]
                        open_rules[max_rule][1] = open_rules[key][1].copy()
                        open_rules[max_rule][1].append([cols[feature], 1])
                        open_rules[max_rule][0] = right
                        # Left
                        open_rules[key][1].append([cols[feature], 0])
                        open_rules[key][0] = left
                    else:
                        open_rules[key][1].append([self.consequent, target])
                        rules.append(open_rules[key][1])
                        open_rules.pop(key)
            tree_rules.append(rules)
        return tree_rules

    def _get_no_dummies_rules(self, tree_rules: list) -> list:
        """Decode dummies back to categories.

        Parameters
        ----------
        tree_rules : list
            Classification rules with dummies.

        Returns
        -------
        list
            Classification rules with categories.
        """
        no_dummies_rules = []
        for rule_pack in tree_rules:
            for rule in rule_pack:
                new_rule = {}
                for part in rule[:-1]:
                    for key, possible_values in self.possible_values_dict.items():
                        if part[0].startswith(key + "_"):
                            for possible_value in possible_values:
                                if part[0].endswith("_" + str(possible_value)):
                                    if part[1] == 1:
                                        if key not in new_rule.keys():
                                            new_rule[key] = []
                                        new_rule[key].append(possible_value)
                                    else:
                                        if key not in new_rule.keys():
                                            new_rule[key] = possible_values.copy()
                                        if possible_value in new_rule[key]: #---------------------
                                            new_rule[key].remove(possible_value)
                new_rule[rule[-1][0]] = [rule[-1][1]]
                no_dummies_rules.append(new_rule)
        return no_dummies_rules

    @staticmethod
    def _get_divided_rules(no_dummies_rules: list) -> list:
        """If more categories from the same attribute is in the rule, more classification rules are derived.

        Parameters
        ----------
        no_dummies_rule : list
            Classification rules with multiple values per attribute.

        Returns
        -------
        list
            Classification rules with a single value per attribute.
        """
        divided_rules = []
        for new_rule in no_dummies_rules:  # new rule
            part_rule = [{}]
            for key, value in new_rule.items():  # values of an attribute
                part_list = []
                for val in value:  # values
                    for part in part_rule:
                        part_dict = part.copy()  # copy the current part of the rule
                        if val != "nan":
                            part_dict[key] = val
                            part_list.append(part_dict)
                part_rule = part_list
            divided_rules += part_rule
        return divided_rules

    def _get_rules_frame(self, divided_rules: list) -> pd.DataFrame:
        """Saves classification rules to data frame.

        Parameters
        ----------
        divided_rules : list
            Classification rules with multiple values per attribute.

        Returns
        -------
        pd.DataFrame
            Classification rules data frame.
        """
        cols = self.antecedent + [self.consequent]
        rules_frame = pd.DataFrame(columns=cols)
        for rule in divided_rules:
            rules_frame = rules_frame.append(rule, ignore_index=True)
        return rules_frame

    def _get_supp_conf(self, divided_rules: list) -> list:
        """Get support and confidence for all classification rules.

        Parameters
        ----------
        divided_rules : list
            List of classification rules.

        Returns
        -------
        list
            Support and confidence.
        """
        supp_conf = []
        all_len = len(self.data)
        consequent = ""
        for rule in divided_rules:
            df_s = self.data.copy()
            for key, value in rule.items():
                if key != self.consequent:
                    df_s = df_s.loc[(df_s[key] == value)]
                else:
                    consequent = value
            antec = len(df_s)
            df_s = df_s.loc[(df_s[self.consequent] == consequent)]
            succ = len(df_s)
            if all_len > 0:
                support = succ / all_len
            else:
                support = None
            if antec > 0:
                confidence = succ / antec
            else:
                confidence = None
            supp_conf.append([support, confidence])
        return supp_conf

    def _set_frame_supp_conf(self, rules_frame: pd.DataFrame, supp_conf: list):
        """Adds support and confidence to data frame.

        Parameters
        ----------
        rules_frame : pd.DataFrame
            Classification rules data frame.
        supp_conf : list
            Support and confidence.
        """
        support = []
        confidence = []
        for values in supp_conf:
            support.append(values[0])
            confidence.append(values[1])
        data_support = pd.DataFrame({'support': support})
        data_confidence = pd.DataFrame({'confidence': confidence})
        rules_frame = rules_frame.join(data_support)
        rules_frame = rules_frame.join(data_confidence)
        self.rules_frame = rules_frame

    def fit(self, antecedent: List[str], consequent: str, supp: float, conf: float, n_estimators: int=30, **kwargs):
        """Train the model.

        It also reduced classification rules by minimal support and minimal confidence.

        Parameters
        ----------
        antecedent : List[str]
            List of antecedent columns names.
        consequent : str
            Consequent column name.
        supp : float
            Minimal support. For example 10 means 10%.
        conf : float
            Minimal confidence .For example 80 means 80%.
        n_estimators : int=30
            Number of generated trees.
        **kwargs :
            Arbitrary keyword arguments for RandomForestRegressor.
        """
        self._prepare_data(antecedent, consequent)
        self._set_dummies()
        rf = self._get_random_forest(n_estimators, **kwargs)
        trees = self._get_trees(rf)
        tree_rules = self._get_tree_rules(trees)
        no_dummies_rules = self._get_no_dummies_rules(tree_rules)
        divided_rules = self._get_divided_rules(no_dummies_rules)
        rules_frame = self._get_rules_frame(divided_rules)
        supp_conf = self._get_supp_conf(divided_rules)
        self._set_frame_supp_conf(rules_frame, supp_conf)
        self.rules_frame = self.rules_frame.drop_duplicates()
        self.rules_frame = self.rules_frame[self.rules_frame.confidence >= (conf / 100)]
        self.rules_frame = self.rules_frame[self.rules_frame.support >= (supp / 100)]
        self.rules_frame = self.rules_frame.reset_index(drop=True)

    def get_frame(self):
        return self.rules_frame
