"""
SIH26091 — Small Enterprise Tax & Investment Advisory
======================================================
Input  : village/location name, margin capital, business sector
Output : Module 2 — Dynamic Financial Statement & Loan Structure
         Module 1 — Location-Aware Tax & Investment Advisory (RAG + Gemini)

The Mahesana_Population.csv dataset is used INTERNALLY to profile the economic
archetype, priority lending eligibility, and consumer purchasing capacity
of the user's location. This profile dynamically drives the area-specific
interest rate, government subsidies, financial statement, and tax calculations.
"""

import sys
import os

# Ensure Windows terminal outputs UTF-8 (handles Rupee symbol ₹ without charmap errors)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from financial_calculator import build_financial_plan
from local_data import find_village, derive_market_profile
from rag_engine import build_retriever, generate_advisory_report


def run(village_name: str, margin_capital: float, sector: str):
    print("=" * 78)
    print(f"  ENTERPRISE FINANCIAL & TAX ADVISORY - Location: {village_name}")
    print(f"  Margin Capital: Rs.{margin_capital:,.0f} | Sector: {sector.title()}")
    print("=" * 78)

    # 1. Location Lookup & Market Profile (silent internal derivation)
    village_row = find_village(village_name)
    market_profile = derive_market_profile(village_row, sector) if village_row else {}

    # 2. Dynamic Financial Statement & Loan Plan
    plan = build_financial_plan(margin_capital, sector=sector, market_profile=market_profile)

    print("\n+----------------------------------------------------------------------------+")
    print(f"| 1. DYNAMIC FINANCIAL & LOAN STATEMENT                                      |")
    print("+----------------------------------------------------------------------------+")
    print(f"  Scheme Selected      : {plan.scheme_name}")
    print(f"  Prevailing Rate      : {plan.interest_rate_annual}% p.a. (Area/Sector Calibrated)")
    print(f"  Total Project Cost   : Rs.{plan.project_cost:,.2f}")
    print(f"  Promoter Margin (10%): Rs.{plan.margin_capital:,.2f}")
    print(f"  Sanctioned Loan      : Rs.{plan.loan_amount:,.2f}")
    print(f"  Eligible Subsidy     : Rs.{plan.subsidy_amount:,.2f}")
    print(f"  Effective Net Loan   : Rs.{plan.effective_loan:,.2f}")
    print(f"  Tenure & Moratorium  : {plan.tenure_years} Years (Moratorium: {plan.moratorium_months} Months)")
    print(f"  Quarterly EMI        : Rs.{plan.quarterly_emi:,.2f}")
    print(f"  Total Interest       : Rs.{plan.total_interest_payable:,.2f}")
    print("+----------------------------------------------------------------------------+")
    print(f"| 2. PROJECTED ANNUAL PROFIT & LOSS (P&L) STATEMENT                          |")
    print("+----------------------------------------------------------------------------+")
    print(f"  Projected Revenue    : Rs.{plan.projected_annual_revenue:,.2f}")
    print(f"  Cost of Goods (COGS) : Rs.{plan.cogs_amount:,.2f}")
    print(f"  Gross Profit         : Rs.{plan.gross_profit:,.2f} ({plan.gross_margin_pct}% Margin)")
    print(f"  Operating Expenses   : Rs.{plan.operating_expenses:,.2f}")
    print(f"  EBITDA (Cash Profit) : Rs.{plan.ebitda:,.2f}")
    print(f"  Annual Debt Service  : Rs.{plan.annual_debt_service:,.2f} (EMI Coverage: {plan.dscr}x DSCR)")
    print(f"  Break-Even Revenue   : Rs.{plan.break_even_revenue:,.2f}")
    print(f"  Net Profit (Pre-Tax) : Rs.{plan.net_profit_before_tax:,.2f}")
    print("+----------------------------------------------------------------------------+")
    print(f"| 3. ESTIMATED ANNUAL TAX LIABILITY                                          |")
    print("+----------------------------------------------------------------------------+")
    print(f"  GST Regime           : {plan.gst_regime}")
    print(f"  Estimated Annual GST : Rs.{plan.estimated_annual_gst:,.2f}")
    print(f"  Income Tax Scheme    : {plan.income_tax_regime}")
    print(f"  Presumptive Income   : Rs.{plan.presumptive_income:,.2f}")
    print(f"  Estimated Income Tax : Rs.{plan.estimated_annual_income_tax:,.2f}")
    print(f"  TOTAL ANNUAL TAX     : Rs.{plan.total_annual_tax:,.2f}")
    print(f"  Net Profit After Tax : Rs.{plan.net_profit_after_tax:,.2f}")
    print(f"  Projected Margin ROI : {plan.return_on_margin_pct}% per annum")
    print("+----------------------------------------------------------------------------+")

    # 3. RAG Knowledge Retrieval & Gemini Advisory Generation
    retriever = build_retriever()
    query = (f"{sector} small enterprise tax investment scheme "
             f"Gujarat rural micro-entrepreneur financial advisory")
    retrieved = retriever.retrieve(query, top_k=4)

    report = generate_advisory_report(
        plan=plan,
        market_profile=market_profile,
        location=village_name,
        sector=sector,
        retrieved_chunks=retrieved,
    )

    print("\n" + "=" * 78)
    print("  GEMINI AI TAX & INVESTMENT ADVISORY REPORT")
    print("=" * 78)
    print(report)


if __name__ == "__main__":
    # Parameters:
    # 1. village_name : Any village/town in Mahesana (e.g. "Visnagar", "Sudasana", "Chelana", "Kheralu", "Vadnagar")
    # 2. margin_capital: Promoter's capital contribution in INR (e.g. 50000, 100000, 200000)
    # 3. sector        : Business sector (e.g. "textiles", "dairy", "retail", "agriculture", "handicraft")
    
    run(village_name="Kheralu", margin_capital=150_000, sector="agriculture")
