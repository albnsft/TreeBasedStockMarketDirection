import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, brier_score_loss, roc_auc_score, f1_score

from itertools import product
import random

from env.FinancialEnvironment import FinancialEnvironment
from utils.utils import Set, WalkForward
import pickle
import os
from pylab import plt, mpl
from env.utils import plot_final
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['font.family'] = 'serif'


class Classifier:
    def __init__(
            self,
            env: FinancialEnvironment = None,
            cv: WalkForward = None,
            model_type: str = None,
            test_dates: Set = None,
            verbose: bool = False,
            seed: int = 42,
            n_iter_opt: int = None,
    ):
        self.env = env
        self.cv = cv
        self.model_type = model_type
        self.test_dates = test_dates
        self.verbose = verbose
        self.seed = seed
        self.n_iter_opt = n_iter_opt
        self.params = {'RF': {}, 'LGB': {}, 'XGB': {}}
        self.init()

    def init(self):
        if self.model_type not in self.params.keys():
            raise Exception(f'{self.model_type} not been implemented')

        base = os.path.dirname(os.path.abspath(__file__))
        path_params = os.path.join(base, 'savings', self.env.ticker, self.model_type)
        if not os.path.exists(path_params): os.makedirs(path_params)
        self.path_params = os.path.join(path_params, f'{str(self.env.step_size.days)}d_params.pkl')
        if os.path.exists(self.path_params):
            with open(self.path_params, "rb") as file:
                self.params[self.model_type] = pickle.load(file)
            print(f'Best hyper parameters: {self.params[self.model_type]}')
        path_results = os.path.join(base.replace('\models', ''), 'results', self.env.ticker, self.model_type)
        if not os.path.exists(path_results): os.makedirs(path_results)
        self.path_results = os.path.join(path_results, f'{str(self.env.step_size.days)}d_results.png')

    def learn(self, test_info: bool=False):
        if not self.params[self.model_type]:
            self.tuning()
        test_from_cv = True if self.test_dates is None else False
        _, _, _, self.y_val, self.y_test = self.fit_predict(test=test_from_cv, verbose=self.verbose)
        if not test_from_cv:
            X, y = self.env.subsample(self.test_dates.start, self.test_dates.end)
            _, self.y_test = self.predict(X, y, [])
        self.reports()
        if test_info:
            return self.report_test['outperformance'].values[0]

    def tuning(self):
        params, product_params, dict_param = self.combination_params()
        random_keys = self.generate_keys(product_params, dict_param)
        """
        Instantiate DF to save the results
        """
        results = pd.DataFrame(columns=['mean_f1score', 'std_f1score', 'total_perf', 'f1_mix_perf'] + list(params), index=range(len(random_keys)))
        best_val_score = -np.inf
        best_params = None
        """
        Finding best hyperparameters over number of trials/combination
        Sorted accorded maximum validation f1score
        Saving values for each trial of hyperparameters set
        """
        for i, key in enumerate(random_keys):
            params = dict_param[key]
            mean_score, std_score, aperf, y_val, _ = self.fit_predict(params, test=False)
            myscore = mean_score*0.95 + aperf*0.05
            if myscore > best_val_score:
                best_val_score = myscore
                best_params = params
            results.iloc[i] = [mean_score, std_score, aperf, myscore] + list(params.values())
        results = results.sort_values('mean_f1score', ascending=False)[:5]
        print(f'Top 5 models leading to the max F1-score on validation set with {self.cv.n_walks} time series split: ')
        print(results.to_string())
        self.params[self.model_type] = best_params
        print(f'Best hyper parameters: {best_params}')
        with open(self.path_params, "wb") as file:
            pickle.dump(self.params[self.model_type], file)

    def fit_predict(self, params: dict = None, test: bool = False, verbose: bool = False):
        if params is not None: self.params[self.model_type] = params
        self.model = self.models[self.model_type]
        y_validations, y_tests = [], []
        for idx, walk in self.cv.get_walks():
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.env.sample(walk, test=test)
            self.model.fit(X_train.values, y_train.values.ravel())
            y_validations, y_val = self.predict(X_val, y_val, y_validations)
            if test:
                y_tests, y_test = self.predict(X_test, y_test, y_tests)
            if verbose:
                pass
                #self.print_res(y_val, y_test)
        all_f1 = list(map(lambda df: self.score(df), y_validations))
        all_aperf = list(map(lambda df: self.performance(df)[-1], y_validations))
        all_y_valid = pd.concat(y_validations)
        all_y_test = pd.concat(y_tests) if test else None
        if verbose: print(pd.DataFrame(self.model['clf'].feature_importances_, index=self.env.X.columns, columns=['Features importance']).to_string())
        return np.mean(all_f1), np.std(all_f1, ddof=1), np.sum(all_aperf), all_y_valid, all_y_test

    def predict(self, X: pd.DataFrame, y: pd.DataFrame, ys: list):
        y['proba'] = self.model.predict_proba(X.values)[:, 1]
        y['predicted'] = np.where(y['proba'] > 0.5, 1, 0)
        ys.append(y)
        return ys, y

    def reports(self):
        self.report_val, self.confusion_val, self.strat_val = self.report(self.y_val, 'validation')
        self.report_test, self.confusion_test, self.strat_test = self.report(self.y_test, 'test')
        if self.verbose:
            plot_final(self.env.ticker, self.model_type, self.env.step_size, self.report_val, self.confusion_val, self.strat_val,
                       self.report_test, self.confusion_test, self.strat_test, self.path_results)

    def combination_params(self):
        """
        Combinations of all hyper parameters
        """
        params, values = zip(*self.space_params[self.model_type].items())
        product_params = list(product(*values))
        dict_param = dict(map(lambda i: (i, dict(zip(params, product_params[i]))),
                              range(len(product_params))))
        return params, product_params, dict_param

    def generate_keys(self, product_params: list, dict_param: dict):
        """
        Defining max number of hyperoptimization trials
        """
        if self.n_iter_opt is not None:
            random.seed(42)
            random_keys = random.sample(list(dict_param), self.n_iter_opt)
        else:
            random_keys = list(range(len(product_params)))
        return random_keys

    def report(self, y: pd.DataFrame, type_env: str):
        #classification metric
        rept_base = classification_report(y['label'], y['predicted'], target_names=['short', 'long'], output_dict=True, zero_division=1)
        rept = pd.DataFrame.from_dict(rept_base['weighted avg'], orient='index').T
        rept['accuracy'] = rept_base['accuracy']
        rept['brier'] = brier_score_loss(y['label'], y['proba'])
        rept['auc'] = roc_auc_score(y['label'], y['proba'])
        confusion = confusion_matrix(y['label'], y['predicted'])
        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['short', 'long'])
        #financial metric
        print('****************************************************************************************************')
        print('***************** On all CV stacked ****************')
        strat, operf, aperf = self.performance(y, type_env, self.verbose)
        rept['aperf_%'] = aperf*100
        rept['operf_%'] = operf*100
        return rept, cm_display, strat

    def score(self, y: pd.DataFrame):
        """
        In our context, we use F1 score as the evaluation metric to optimize the model's performance.
        This is because both false positives and false negatives can have significant financial consequences
        and the F1 score balances the trade-off between precision and recall to provide a better overall measure
        of the model's performance.
        """
        return f1_score(y['label'], y['predicted'], average='weighted')

    def performance(self, y: pd.DataFrame, type_env: str = 'validation', verbose: bool = False):
        strat, operf, aperf = self.env.backtest(y, type_env=type_env, type_algo=self.model_type, verbose=verbose, f1=self.score)
        return strat, operf, aperf

    def print_res(self, val: pd.DataFrame, test: pd.DataFrame):
        self.performance(val, 'validation', True)
        if test is not None: self.performance(test, 'test', True)

    @property
    def models(self):
        return {
            'RF': Pipeline([
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('clf', RandomForestClassifier(**self.params['RF'], random_state=self.seed, n_jobs=-1, class_weight="balanced_subsample"))
            ]),
            'LGB': Pipeline([
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('clf', lgb.LGBMClassifier(**self.params['LGB'], random_state=self.seed, n_jobs=-1))
            ]),
            'XGB': Pipeline([
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('clf', xgb.XGBClassifier(**self.params['XGB'], random_state=self.seed, n_jobs=-1))
            ]),
        }

    @property
    def space_params(self):
        return {
            'RF': {
                'n_estimators': [200, 300, 500],
                'max_features': [0.75, 0.9, 1.0],
                'max_samples': [0.75, 0.9, 1.0],
            },
            'LGB': {
                'max_depth': [4, 6, 8, 10],
                'subsample': [0.25, 0.5, 0.75],#
                'n_estimators': [100, 200, 300],
                'feature_fraction': [0.25, 0.5, 0.75],#
                'learning_rate': [0.001, 0.01, 0.1, 1],
            },
            'XGB': {
                'max_depth': [4, 6, 8, 10],
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.001, 0.01, 0.1],
            },
        }
