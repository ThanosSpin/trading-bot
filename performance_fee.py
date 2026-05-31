# fee_model.py

import argparse

def calculate_fees(
    starting_capital,
    annual_return_pct,
    resource_fee_pct=2.0,
    hurdle_pct=5.0,
    performance_fee_pct=15.0
):
    """
    Calculate annual fees and ending capital.

    starting_capital: float ($)
    annual_return_pct: float (%)
    resource_fee_pct: float (% of AUM)
    hurdle_pct: float (% return)
    performance_fee_pct: float (% of excess profit)
    """

    # Gross profit
    gross_profit = starting_capital * (annual_return_pct / 100.0)
    gross_value = starting_capital + gross_profit

    # Resource fee (e.g. 2% AUM)
    resource_fee = starting_capital * (resource_fee_pct / 100.0)

    # Capital after resource fee
    value_after_resource_fee = gross_value - resource_fee

    # Profit after resource fee
    net_profit_after_resource = value_after_resource_fee - starting_capital

    # Hurdle profit
    hurdle_profit = starting_capital * (hurdle_pct / 100.0)

    # Performance fee on excess above hurdle
    feeable_profit = max(0.0, net_profit_after_resource - hurdle_profit)
    performance_fee = feeable_profit * (performance_fee_pct / 100.0)

    # Final values
    ending_capital = value_after_resource_fee - performance_fee
    total_fees = resource_fee + performance_fee

    return {
        "starting_capital": starting_capital,
        "annual_return_pct": annual_return_pct,
        "gross_profit": gross_profit,
        "resource_fee": resource_fee,
        "performance_fee": performance_fee,
        "total_fees": total_fees,
        "ending_capital": ending_capital,
    }


def run_scenarios(
    starting_capital,
    return_scenarios,
    resource_fee_pct=2.0,
    hurdle_pct=5.0,
    performance_fee_pct=15.0,
):
    print(f"\nStarting Capital: ${starting_capital:,.2f}")
    print(f"Resource fee:     {resource_fee_pct:.2f}% of AUM")
    print(f"Hurdle rate:      {hurdle_pct:.2f}%")
    print(f"Performance fee:  {performance_fee_pct:.2f}% of excess\n")
    print("-" * 70)

    for r in return_scenarios:
        result = calculate_fees(
            starting_capital,
            r,
            resource_fee_pct=resource_fee_pct,
            hurdle_pct=hurdle_pct,
            performance_fee_pct=performance_fee_pct,
        )

        print(f"Annual Return: {r}%")
        print(f"  Gross Profit:        ${result['gross_profit']:,.2f}")
        print(f"  Resource Fee:        ${result['resource_fee']:,.2f}")
        print(f"  Performance Fee:     ${result['performance_fee']:,.2f}")
        print(f"  Total Fees:          ${result['total_fees']:,.2f}")
        print(f"  Ending Capital:      ${result['ending_capital']:,.2f}")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance fee scenarios")
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital in dollars (default: 100000)",
    )
    parser.add_argument(
        "--resource-fee",
        type=float,
        default=2.0,
        help="Resource fee as %% of AUM (default: 2.0)",
    )
    parser.add_argument(
        "--hurdle",
        type=float,
        default=5.0,
        help="Hurdle rate in %% (default: 5.0)",
    )
    parser.add_argument(
        "--perf-fee",
        type=float,
        default=15.0,
        help="Performance fee as %% of excess profit (default: 15.0)",
    )
    parser.add_argument(
        "--returns",
        type=float,
        nargs="*",
        default=[-5, 0, 4, 8, 12, 15, 20, 30],
        help="List of annual return scenarios in % (default: -5 0 4 8 12 15 20 30)",
    )

    args = parser.parse_args()

    run_scenarios(
        starting_capital=args.capital,
        return_scenarios=args.returns,
        resource_fee_pct=args.resource_fee,
        hurdle_pct=args.hurdle,
        performance_fee_pct=args.perf_fee,
    )