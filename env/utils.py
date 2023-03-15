from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from env.FinancialEnvironment import FinancialEnvironment


from pylab import plt
import numpy as np


def plot_final(
        ticker,
        algo_name,
        frequency,
        report_val,
        confusion_val,
        strat_val,
        report_test,
        confusion_test,
        strat_test,
        path,
):

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    name = f"{algo_name} - {ticker} - {frequency.days}days"
    plt.suptitle(name)

    columns = ['accuracy', 'recall', 'precision', 'f1-score', 'brier', 'auc', 'aperf_%', 'operf_%']
    done_info = np.round(report_val[columns], 2)
    done_info_eval = np.round(report_test[columns], 2)
    print(done_info_eval.to_string())

    table = ax_dict["A"].table(
        cellText=done_info.values,
        colLabels=done_info.columns,
        loc="center",
    )
    table.set_fontsize(10)
    #table.scale(0.5, 1.1)
    ax_dict["A"].set_axis_off()
    ax_dict["A"].title.set_text("Results validation")

    table = ax_dict["B"].table(
        cellText=done_info_eval.values,
        colLabels=done_info_eval.columns,
        loc="center",
    )
    table.set_fontsize(10)
    #table.scale(0.5, 1.1)
    ax_dict["B"].set_axis_off()
    ax_dict["B"].title.set_text("Results testing")

    confusion_val.plot(ax=ax_dict["C"])
    confusion_test.plot(ax=ax_dict["D"])

    strat_val[['market', 'strategy']].plot(ax=ax_dict["E"], ylabel='$',
                      title=f'{algo_name} vs Market gross performance + entry position')
    short_ticks = []
    long_ticks = []
    last_position = None
    for i, tick in enumerate(strat_val.index):
        if strat_val['position'].iloc[i] == -1 and last_position != -1:
            short_ticks.append(tick)
            last_position = -1
        elif strat_val['position'].iloc[i] == 1 and last_position != 1:
            long_ticks.append(tick)
            last_position = 1
    ax_dict["E"].plot(short_ticks, strat_val['market'].loc[short_ticks].values, 'ro')
    ax_dict["E"].plot(long_ticks, strat_val['market'].loc[long_ticks].values, 'go')

    strat_test[['market', 'strategy']].plot(ax=ax_dict["F"], ylabel='$',
                      title=f'{algo_name}  vs Market gross performance + entry position')
    short_ticks = []
    long_ticks = []
    last_position = None
    for i, tick in enumerate(strat_test.index):
        if strat_test['position'].iloc[i] == -1 and last_position != -1:
            short_ticks.append(tick)
            last_position = -1
        elif strat_test['position'].iloc[i] == 1 and last_position != 1:
            long_ticks.append(tick)
            last_position = 1
    plt.plot(short_ticks, strat_test['market'].loc[short_ticks].values, 'ro')
    plt.plot(long_ticks, strat_test['market'].loc[long_ticks].values, 'go')


    # Write plot to pdf
    fig.savefig(f'{path}')
    plt.close(fig)