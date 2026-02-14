# fee_model.py

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
    gross_profit = starting_capital * (annual_return_pct / 100)
    gross_value = starting_capital + gross_profit

    # Resource fee (2% AUM)
    resource_fee = starting_capital * (resource_fee_pct / 100)

    # Capital after resource fee
    value_after_resource_fee = gross_value - resource_fee

    # Profit after resource fee
    net_profit_after_resource = value_after_resource_fee - starting_capital

    # Hurdle profit
    hurdle_profit = starting_capital * (hurdle_pct / 100)

    # Performance fee
    feeable_profit = max(0, net_profit_after_resource - hurdle_profit)
    performance_fee = feeable_profit * (performance_fee_pct / 100)

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
        "ending_capital": ending_capital
    }


def run_scenarios(starting_capital, return_scenarios):
    print(f"\nStarting Capital: ${starting_capital:,.2f}")
    print("-" * 70)

    for r in return_scenarios:
        result = calculate_fees(starting_capital, r)

        print(f"Annual Return: {r}%")
        print(f"  Gross Profit:        ${result['gross_profit']:,.2f}")
        print(f"  Resource Fee (2%):   ${result['resource_fee']:,.2f}")
        print(f"  Performance Fee:    ${result['performance_fee']:,.2f}")
        print(f"  Total Fees:         ${result['total_fees']:,.2f}")
        print(f"  Ending Capital:     ${result['ending_capital']:,.2f}")
        print("-" * 70)


if __name__ == "__main__":
    # Example usage
    capital = 100_000

    scenarios = [
        -5,
        0,
        4,
        8,
        12,
        15,
        20,
        30
    ]

    run_scenarios(capital, scenarios)