from features.Features import *
from database import HistoricalDatabase
from utils.utils import WalkForward


class FinancialEnvironment:
    def __init__(
            self,
            database: HistoricalDatabase = None,
            features: list = None,
            ticker: str = None,
            step_size: timedelta = None,
            normalisation_on: bool = False,
    ):

        self.database = database
        self.ticker = ticker
        self.step_size = step_size
        self.start_of_trading = database.start_date[ticker]
        self.end_of_trading = database.end_date[ticker]
        self.features = features or self.get_default_features(step_size, normalisation_on)
        self.max_feature_window_size = max([feature.window_size for feature in self.features])
        self.state: State = None
        self._check_params()
        self.init()

    def init(self) -> np.ndarray:
        now_is = self.start_of_trading  # (self.max_feature_window_size + self.step_size * self.n_lags_feature)
        self.state = State(market=self._get_market_data(now_is), now_is=now_is)
        data, dates = [], [self.state.now_is]
        while self.end_of_trading >= self.state.now_is:
            self._forward()
            data.append(self.get_features())
            dates.append(self.state.now_is)
        self.data = pd.DataFrame(data, index=dates[:-1], columns=[feature.name for feature in self.features]).dropna()
        self.data = self.data[self.data['Direction']!=0] #for binary classification, 0 being outsider
        self.y = pd.DataFrame(np.where(self.data['Direction'].iloc[1:]==1, 1, 0), columns=['label'], index=self.data.iloc[1:].index)
        self.X = self.data.drop(columns=['Direction']).shift(1).iloc[1:]


    def _forward(self):
        self._update_features()
        self.update_internal_state()

    def _update_features(self):
        for feature in self.features:
            feature.update(self.state)

    def get_features(self) -> np.ndarray:
        return np.array([feature.current_value for feature in self.features])

    def update_internal_state(self):
        self.state.now_is += self.step_size
        if self.state.now_is not in self.database.calendar[self.ticker] and self.state.now_is <= self.end_of_trading:
            self.state.now_is = self.database.get_next_timestep(self.state.now_is, self.ticker)
        self.state.market = self._get_market_data(self.state.now_is)

    def _get_market_data(self, datepoint: timedelta):
        data = self.database.get_last_snapshot(datepoint, self.ticker)
        return Market(**{k.lower(): v for k, v in data.to_dict().items()})

    def _check_params(self):
        assert self.start_of_trading <= self.end_of_trading, "Start of trading Nonsense"

    def subsample(self, start, end):
        return (self.X.loc[start: end], self.y.loc[start: end])

    def sample(self, walk: WalkForward, test: bool = False):
        if not test:
            return self.subsample(walk.train.start, walk.train.end), \
                   self.subsample(walk.valid.start, walk.valid.end), \
                   (None, None)
        else:
            return self.subsample(walk.train.start, walk.train.end), \
                   self.subsample(walk.valid.start, walk.valid.end), \
                   self.subsample(walk.test.start, walk.test.end)


    @staticmethod
    def get_default_features(step_size: timedelta, normalisation_on: bool):
        return [Direction(update_frequency=step_size,
                          normalisation_on=normalisation_on),
                RSI(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                STOCH(update_frequency=step_size,
                      normalisation_on=normalisation_on),
                WilliamsR(update_frequency=step_size,
                      normalisation_on=normalisation_on),
                MACD(update_frequency=step_size,
                     normalisation_on=normalisation_on),
                PROC(update_frequency=step_size,
                    normalisation_on=normalisation_on),
                OBV(update_frequency=step_size,
                    normalisation_on=normalisation_on)]

    def backtest(self, y: pd.DataFrame = None, spread: float = 0.00006, cash: float = 100,
                 type_env: str = 'valid', verbose: bool = True, type_algo: str = 'rf', f1 = None):

        calendar = y.index
        predictions = y['predicted']
        f1 = f1(y)

        data = self.database.data[self.ticker]['Close'].to_frame().loc[calendar].copy()
        data['return'] = data.Close.pct_change().fillna(0)
        data['position'] = np.where(predictions > 0.5, 1, -1)

        def ptf_base(start_sum, rets):
            v = start_sum
            for r in rets:
                v = v * (1 + r)
                yield v

        data['strategy'] = data['position'] * data['return'] # Calculates the strategy returns given the position values
        # determine when a trade takes place
        trades = data['position'].diff().fillna(1) != 0
        # instantiate strategy with transaction cost
        data['strategy_tc'] = data['strategy']
        # spread = 0.00006 --> bid-ask spread on professional level
        tc = spread / data.Close.mean()
        # subtract transaction costs from return when trade takes place
        data['strategy_tc'][trades] -= tc
        # compute the VL base 100 of the passive returns, strategy and strategy with transaction cost
        data['market'] = pd.Series(list(ptf_base(cash, data['return'])), index=data.index)
        #data['cum_strategy'] = pd.Series(list(ptf_base(cash, data['strategy'])), index=data.index)
        data['strategy'] = pd.Series(list(ptf_base(cash, data['strategy_tc'])), index=data.index)
        VL_strat = data['strategy'].iloc[-1]
        aperf = VL_strat / cash - 1
        operf = aperf - (data['market'].iloc[-1] / cash - 1)
        if verbose:
            print(f'************************ On {type_env} set *********************************')
            print(f'The weighted F1-score is {np.round(f1, 3)}')
            print(f'The number of trades is {sum(trades)}, there is a total of {len(data)} ticks')
            print('The absolute performance of the strategy with tc is {:.1%}'.format(aperf))
            print('The outperformance of the strategy with tc is {:.1%}'.format(operf))
            print(100 * '*')
        plot = data[['market', 'strategy', 'position']]
        return plot, operf, aperf