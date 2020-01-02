import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List


class RandomForest:
    """
    Check all classification couples if they can make action rule
    """

    def __init__(self):
        """
        Initialise by reduced tables.
        """
        self.data = pd.DataFrame()
        self.data_dummies = pd.DataFrame()
        self.antecedents = []
        self.consequent = ""
        self.possible_values_dict = {}
        self.rules_frame = pd.DataFrame()

    def read_csv(self, file: str, **kwargs):
        """
        Load data from csv.
        """
        self.data = pd.read_csv(file, **kwargs)

    def load_pandas(self, data_frame: pd.DataFrame):
        """
        Load data from pandas dataframe.
        """
        self.data = data_frame

    def prepare_data(self, antecedents: List[str], consequent: str):
        """
        Prepare data.
        """
        reduced = antecedents + [consequent]
        self.data = self.data[reduced]
        self.data = self.data.fillna("nan")
        self.antecedents = antecedents
        self.consequent = consequent
        for col in self.antecedents:
            self.possible_values_dict[col] = self.data[col].unique().tolist()

    def set_dummies(self):
        """
        Get dummies.
        """
        data_reduced = self.data[self.antecedents]
        self.data_dummies = pd.get_dummies(data_reduced, columns=self.antecedents)

    def get_random_forest(self, n_estimators: int=30, **kwargs):
        rf = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        train_x = self.data_dummies
        train_y = self.data[self.consequent].values.ravel()
        rf.fit(train_x, train_y)
        return rf

    @staticmethod
    def get_trees(rf: RandomForestRegressor):
        trees = {}
        for tree_idx, est in enumerate(rf.estimators_):
            trees[tree_idx] = {}
            tree = est.tree_
            assert tree.value.shape[1] == 1  # no support for multi-output
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
                class_idx_val = value[0][0]
                if class_idx_val > 0.5:
                    class_idx = 1
                else:
                    class_idx = 0
                # Tree representation
                if left == -1 and right == -1:
                    trees[tree_idx][node_idx] = [feature, th, left, right, class_idx]
                else:
                    trees[tree_idx][node_idx] = [feature, th, left, right, None]
        return trees

    def get_tree_rules(self, trees: dict):
        tree_rules = []
        cols = self.data_dummies.columns
        for tid, tree in trees.items():
            open_rules = {0: [0, []]}
            rules = []
            max_rule = 0
            while len(open_rules) > 0:
                loop_rules = open_rules.copy()
                for key, value in loop_rules.items():
                    feature, th, left, right, class_idx = tree[value[0]]
                    if class_idx is None:
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
                        open_rules[key][1].append([self.consequent, class_idx])
                        rules.append(open_rules[key][1])
                        open_rules.pop(key)
            tree_rules.append(rules)
        return tree_rules

    def get_no_dummies_rules(self, tree_rules):
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
                                        new_rule[key].remove(possible_value)
                new_rule[rule[-1][0]] = [rule[-1][1]]
                no_dummies_rules.append(new_rule)
        return no_dummies_rules

    @staticmethod
    def get_divided_rules(no_dummies_rules):
        divided_rules = []
        for new_rule in no_dummies_rules:  # nove pravidlo
            part_rule = [{}]
            for key, value in new_rule.items():  # atribut a jeho hodnoty
                part_list = []
                for val in value:  # hodnoty
                    for part in part_rule:
                        part_dict = part.copy()  # kopiruj soucasnou cast pravidla
                        if val != "nan":
                            part_dict[key] = val
                            part_list.append(part_dict)
                part_rule = part_list
            divided_rules += part_rule
        return divided_rules

    def get_rules_frame(self, divided_rules):
        cols = self.antecedents + [self.consequent]
        rules_frame = pd.DataFrame(columns=cols)
        for rule in divided_rules:
            rules_frame = rules_frame.append(rule, ignore_index=True)
        return rules_frame

    def get_supp_conf(self, rules_frame, divided_rules):
        supp_conf = []
        all_len = len(rules_frame)
        consequent = ""
        for rule in divided_rules:
            df_s = rules_frame.copy()
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

    def set_frame_supp_conf(self, rules_frame, supp_conf):
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

    def fit(self, antecedents: List[str], consequent: str, supp: float, conf: float, n_estimators: int=30, **kwargs):
        self.prepare_data(antecedents, consequent)
        self.set_dummies()
        rf = self.get_random_forest(n_estimators, **kwargs)
        trees = self.get_trees(rf)
        tree_rules = self.get_tree_rules(trees)
        no_dummies_rules = self.get_no_dummies_rules(tree_rules)
        divided_rules = self.get_divided_rules(no_dummies_rules)
        rules_frame = self.get_rules_frame(divided_rules)
        supp_conf = self.get_supp_conf(rules_frame, divided_rules)
        self.set_frame_supp_conf(rules_frame, supp_conf)
        self.rules_frame = self.rules_frame.drop_duplicates()
        self.rules_frame = self.rules_frame[self.rules_frame.confidence >= (conf / 100)]
        self.rules_frame = self.rules_frame[self.rules_frame.support >= (supp / 100)]
        self.rules_frame = self.rules_frame.reset_index(drop=True)

    def get_frame(self):
        return self.rules_frame
