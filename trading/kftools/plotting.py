import empyrical as ep
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pyfolio as pf

def create_simple_tear_sheet(returns,
                             positions=None,
                             transactions=None,
                             benchmark_rets=None,
                             slippage=None,
                             estimate_intraday='infer',
                             live_start_date=None,
                             turnover_denom='AGB',
                             header_rows=None,
                             file_name=None):
    """
    A clone of pyfolio function with additional export file_name argument.
    For documentation and arguments see pyfolio.create_simple_tear_sheet(...).
    """

    positions = pf.utils.check_intraday(estimate_intraday, returns,
                                        positions, transactions)

    if (slippage is not None) and (transactions is not None):
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    always_sections = 4
    positions_sections = 4 if positions is not None else 0
    transactions_sections = 2 if transactions is not None else 0
    live_sections = 1 if live_start_date is not None else 0
    benchmark_sections = 1 if benchmark_rets is not None else 0

    vertical_sections = sum([
        always_sections,
        positions_sections,
        transactions_sections,
        live_sections,
        benchmark_sections,
    ])

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    pf.plotting.show_perf_stats(returns,
                                benchmark_rets,
                                positions=positions,
                                transactions=transactions,
                                turnover_denom=turnover_denom,
                                live_start_date=live_start_date,
                                header_rows=header_rows)

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)

    ax_rolling_returns = plt.subplot(gs[:2, :])
    i = 2
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1

    pf.plotting.plot_rolling_returns(returns,
                                     factor_returns=benchmark_rets,
                                     live_start_date=live_start_date,
                                     cone_std=(1.0, 1.5, 2.0),
                                     ax=ax_rolling_returns)
    ax_rolling_returns.set_title('Cumulative returns')

    if benchmark_rets is not None:
        pf.plotting.plot_rolling_beta(returns, benchmark_rets, ax=ax_rolling_beta)

    pf.plotting.plot_rolling_sharpe(returns, ax=ax_rolling_sharpe)

    pf.plotting.plot_drawdown_underwater(returns, ax=ax_underwater)

    if positions is not None:
        # Plot simple positions tear sheet
        ax_exposures = plt.subplot(gs[i, :])
        i += 1
        ax_top_positions = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_holdings = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_long_short_holdings = plt.subplot(gs[i, :])
        i += 1

        positions_alloc = pf.pos.get_percent_alloc(positions)

        pf.plotting.plot_exposures(returns, positions, ax=ax_exposures)

        pf.plotting.show_and_plot_top_positions(returns,
                                                positions_alloc,
                                                show_and_plot=0,
                                                hide_positions=False,
                                                ax=ax_top_positions)

        pf.plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

        pf.plotting.plot_long_short_holdings(returns, positions_alloc,
                                             ax=ax_long_short_holdings)

        if transactions is not None:
            # Plot simple transactions tear sheet
            ax_turnover = plt.subplot(gs[i, :])
            i += 1
            ax_txn_timings = plt.subplot(gs[i, :])
            i += 1

            pf.plotting.plot_turnover(returns,
                                      transactions,
                                      positions,
                                      ax=ax_turnover)

            pf.plotting.plot_txn_time_hist(transactions, ax=ax_txn_timings)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    # fig.tight_layout()
    if file_name:
        fig.savefig(file_name)
    else:
        plt.show()
    plt.close(fig)


def create_returns_tear_sheet(returns, positions=None,
                              transactions=None,
                              live_start_date=None,
                              cone_std=(1.0, 1.5, 2.0),
                              benchmark_rets=None,
                              bootstrap=False,
                              turnover_denom='AGB',
                              header_rows=None,
                              file_name=None):
    """
    A clone of pyfolio function with additional export file_name argument.
    For documentation and arguments see pyfolio.create_returns_tear_sheet(...).
    """

    if benchmark_rets is not None:
        returns = pf.utils.clip_returns_to_benchmark(returns, benchmark_rets)

    pf.plotting.show_perf_stats(returns, benchmark_rets,
                                positions=positions,
                                transactions=transactions,
                                turnover_denom=turnover_denom,
                                bootstrap=bootstrap,
                                live_start_date=live_start_date,
                                header_rows=header_rows)

    pf.plotting.show_worst_drawdown_periods(returns)

    vertical_sections = 11

    if live_start_date is not None:
        vertical_sections += 1
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    if benchmark_rets is not None:
        vertical_sections += 1

    if bootstrap:
        vertical_sections += 1

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])

    i = 2
    ax_rolling_returns_vol_match = plt.subplot(gs[i, :],
                                               sharex=ax_rolling_returns)
    i += 1
    ax_rolling_returns_log = plt.subplot(gs[i, :],
                                         sharex=ax_rolling_returns)
    i += 1
    ax_returns = plt.subplot(gs[i, :],
                             sharex=ax_rolling_returns)
    i += 1
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_volatility = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_drawdown = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_monthly_heatmap = plt.subplot(gs[i, 0])
    ax_annual_returns = plt.subplot(gs[i, 1])
    ax_monthly_dist = plt.subplot(gs[i, 2])
    i += 1
    ax_return_quantiles = plt.subplot(gs[i, :])
    i += 1

    pf.plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)
    ax_rolling_returns.set_title(
        'Cumulative returns')

    pf.plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=(benchmark_rets is not None),
        legend_loc=None,
        ax=ax_rolling_returns_vol_match)
    ax_rolling_returns_vol_match.set_title(
        'Cumulative returns volatility matched to benchmark')

    pf.plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns_log)
    ax_rolling_returns_log.set_title(
        'Cumulative returns on logarithmic scale')

    pf.plotting.plot_returns(
        returns,
        live_start_date=live_start_date,
        ax=ax_returns,
    )
    ax_returns.set_title(
        'Returns')

    if benchmark_rets is not None:
        pf.plotting.plot_rolling_beta(
            returns, benchmark_rets, ax=ax_rolling_beta)

    pf.plotting.plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, ax=ax_rolling_volatility)

    pf.plotting.plot_rolling_sharpe(
        returns, ax=ax_rolling_sharpe)

    # Drawdowns
    pf.plotting.plot_drawdown_periods(
        returns, top=5, ax=ax_drawdown)

    pf.plotting.plot_drawdown_underwater(
        returns=returns, ax=ax_underwater)

    pf.plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    pf.plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    pf.plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    pf.plotting.plot_return_quantiles(
        returns,
        live_start_date=live_start_date,
        ax=ax_return_quantiles)

    if bootstrap and benchmark_rets is not None:
        ax_bootstrap = plt.subplot(gs[i, :])
        pf.plotting.plot_perf_stats(returns, benchmark_rets,
                                    ax=ax_bootstrap)
    elif bootstrap:
        raise ValueError('bootstrap requires passing of benchmark_rets.')

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if file_name:
        fig.savefig(file_name)
    else:
        plt.show()
    plt.close(fig)
