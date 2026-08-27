"""
Module 2: Smart Financial Statement Calculator & Dynamic Scheme Router
-----------------------------------------------------------------------
Determines loan terms, area-specific prevailing interest rates (using Gemini
and local market profiles), and calculates complete projected financial
statements (Revenue, Costs, Gross Margin, Net Profit, Debt-Service Coverage,
and GST/Income Tax liability).
"""

import os
import re
import json
from dataclasses import dataclass


@dataclass
class FinancialStatement:
    # Loan & Capital Structure
    margin_capital: float
    project_cost: float
    loan_amount: float
    subsidy_amount: float
    effective_loan: float
    scheme_name: str
    interest_rate_annual: float
    tenure_years: int
    moratorium_months: int
    quarterly_emi: float
    total_interest_payable: float
    total_repayment: float

    # Projected Annual Financial Statement (P&L)
    projected_annual_revenue: float
    cogs_amount: float
    gross_profit: float
    gross_margin_pct: float
    operating_expenses: float
    ebitda: float
    annual_debt_service: float
    net_profit_before_tax: float
    dscr: float
    break_even_revenue: float

    # Tax Statements
    gst_regime: str
    estimated_annual_gst: float
    income_tax_regime: str
    presumptive_income: float
    estimated_annual_income_tax: float
    total_annual_tax: float
    net_profit_after_tax: float
    return_on_margin_pct: float


def query_gemini_for_interest_rate(market_profile: dict, sector: str, project_cost: float) -> dict:
    """
    Calls Google Gemini to look up current prevailing area interest rates,
    subsidies, and lending guidelines in Gujarat for the given location & sector.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_rate_by_profile(market_profile, sector, project_cost)

    try:
        # pyrefly: ignore [missing-import]
        from google import genai
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model="gemini-3.6-flash")

        loc = market_profile.get("location", "Mahesana, Gujarat")
        loc_type = market_profile.get("location_type", "Rural Village")
        priority = market_profile.get("priority_lending_zone", True)
        econ = market_profile.get("dominant_economy", "Agriculture")

        prompt = f"""You are a bank lending specialist in Gujarat, India.
Determine the current prevailing lending interest rate (% p.a.), government scheme, and subsidy
for a small {sector} enterprise.

Location: {loc} ({loc_type})
Local Economy: {econ}
Priority Sector / Rural Status: {priority}
Project Cost: Rs.{project_cost:,.0f}

Respond ONLY in strict JSON format with no markdown wrappers or other text:
{{
  "interest_rate_annual": <float between 5.5 and 9.5>,
  "scheme_name": "<string: e.g. PMEGP Rural Subsidy Scheme / MUDRA Tarun / Gujarat Cottage Industry / NABARD Dairy Scheme>",
  "subsidy_pct": <float between 15.0 and 35.0>,
  "tenure_years": <int: 3 to 7>,
  "moratorium_months": <int: 3 to 6>,
  "rate_rationale": "<brief 1-sentence reason for this rate in this area>"
}}"""

        res = chat.send_message(prompt)
        text = res.text.strip()
        # Clean JSON if wrapped in backticks
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return data
    except Exception:
        return _fallback_rate_by_profile(market_profile, sector, project_cost)


def _fallback_rate_by_profile(market_profile: dict, sector: str, project_cost: float) -> dict:
    """
    Dynamic deterministic rate determination when Gemini is offline.
    Adapts based on rural vs urban, priority sector, and artisan/agriculture status.
    """
    is_rural = market_profile.get("is_rural", True)
    sc_st_r = market_profile.get("sc_st_ratio_pct", 0)
    sk = sector.strip().lower()

    if sk in ("dairy", "agriculture", "agri") and is_rural:
        rate = 6.5
        scheme = "NABARD / Priority Agri-MSME Scheme"
        subsidy = 30.0 if sc_st_r >= 15 else 25.0
        tenure = 5
        moratorium = 6
        rationale = "Priority rural agricultural lending rate with NABARD refinancing in Mahesana"
    elif sk in ("textile", "textiles", "weaving", "handicraft"):
        rate = 6.25 if is_rural else 7.5
        scheme = "Gujarat Cottage & Khadi Village Industries Scheme (PMEGP)"
        subsidy = 35.0 if is_rural else 25.0
        tenure = 7
        moratorium = 6
        rationale = "Artisan and handloom cluster concessional rate under Gujarat State Textile/MSME policy"
    elif is_rural:
        rate = 7.0
        scheme = "PMEGP Rural Small Enterprise Scheme"
        subsidy = 35.0 if sc_st_r >= 15 else 25.0
        tenure = 5
        moratorium = 3
        rationale = "Rural enterprise subsidized priority lending rate"
    else:
        rate = 8.25
        scheme = "MUDRA / Commercial MSME Term Loan"
        subsidy = 15.0
        tenure = 5
        moratorium = 3
        rationale = "Prevailing semi-urban commercial banking MSME benchmark rate"

    return {
        "interest_rate_annual": rate,
        "scheme_name": scheme,
        "subsidy_pct": subsidy,
        "tenure_years": tenure,
        "moratorium_months": moratorium,
        "rate_rationale": rationale,
    }


def calculate_amortization(loan_amount: float, annual_rate: float, tenure_years: int, moratorium_months: int) -> dict:
    """Calculate quarterly EMI using standard reducing-balance amortization."""
    quarterly_rate = (annual_rate / 100) / 4
    total_quarters = tenure_years * 4
    moratorium_quarters = round(moratorium_months / 3)
    repayment_quarters = max(1, total_quarters - moratorium_quarters)

    if quarterly_rate == 0:
        quarterly_emi = loan_amount / repayment_quarters
    else:
        quarterly_emi = (
            loan_amount
            * quarterly_rate
            * (1 + quarterly_rate) ** repayment_quarters
        ) / ((1 + quarterly_rate) ** repayment_quarters - 1)

    total_repayment = quarterly_emi * repayment_quarters
    total_interest = total_repayment - loan_amount

    return {
        "quarterly_emi": round(quarterly_emi, 2),
        "total_interest": round(total_interest, 2),
        "total_repayment": round(total_repayment, 2),
        "repayment_quarters": repayment_quarters,
    }


def build_financial_plan(margin_capital: float, sector: str = "textiles", market_profile: dict = None) -> FinancialStatement:
    """
    Builds a full dynamic financial statement with P&L, taxes, and loan amortization
    tailored to the specific village/location.
    """
    if margin_capital <= 0:
        raise ValueError("Margin capital must be greater than 0.")

    if market_profile is None:
        market_profile = {}

    # 1. Project Cost (Margin is 10% of total project cost)
    project_cost = round(margin_capital / 0.10, 2)

    # 2. Get dynamic interest rate & scheme from Gemini / Location profile
    rate_info = query_gemini_for_interest_rate(market_profile, sector, project_cost)
    annual_rate = float(rate_info.get("interest_rate_annual", 7.5))
    scheme_name = rate_info.get("scheme_name", "PMEGP / MSME Term Loan")
    subsidy_pct = float(rate_info.get("subsidy_pct", 25.0))
    tenure_years = int(rate_info.get("tenure_years", 5))
    moratorium_months = int(rate_info.get("moratorium_months", 3))

    # 3. Capital structure
    gross_loan = round(project_cost * 0.90, 2)
    subsidy_amount = round(project_cost * (subsidy_pct / 100), 2)
    effective_loan = round(gross_loan - subsidy_amount, 2) if subsidy_amount < gross_loan else gross_loan

    # 4. Amortization
    amort = calculate_amortization(gross_loan, annual_rate, tenure_years, moratorium_months)
    quarterly_emi = amort["quarterly_emi"]
    total_interest = amort["total_interest"]
    total_repayment = amort["total_repayment"]
    annual_debt_service = round(quarterly_emi * 4, 2)

    # 5. Dynamic P&L Financial Projections
    # Capital turnover ratio varies by sector & consumer base size
    c_size = market_profile.get("consumer_base_size", "Medium")
    turnover_mult = 2.4 if c_size == "Large" else (2.0 if c_size == "Medium" else 1.6)
    
    sk = sector.strip().lower()
    if sk in ("retail", "trading", "shop"):
        cogs_ratio = 0.70
        opex_ratio = 0.12
    elif sk in ("dairy", "agriculture"):
        cogs_ratio = 0.55
        opex_ratio = 0.18
    elif sk in ("textile", "textiles", "weaving", "handicraft"):
        cogs_ratio = 0.50
        opex_ratio = 0.20
    else:
        cogs_ratio = 0.58
        opex_ratio = 0.16

    projected_annual_revenue = round(project_cost * turnover_mult, 2)
    cogs_amount = round(projected_annual_revenue * cogs_ratio, 2)
    gross_profit = round(projected_annual_revenue - cogs_amount, 2)
    gross_margin_pct = round((gross_profit / projected_annual_revenue) * 100, 1)

    operating_expenses = round(projected_annual_revenue * opex_ratio, 2)
    ebitda = round(gross_profit - operating_expenses, 2)
    net_profit_before_tax = round(ebitda - annual_debt_service, 2)
    dscr = round(ebitda / annual_debt_service, 2) if annual_debt_service > 0 else 0

    # Break-even Revenue = (Fixed Opex + Debt Service) / Gross Margin Ratio
    gross_margin_ratio = (gross_profit / projected_annual_revenue) if projected_annual_revenue > 0 else 0.3
    fixed_annual_costs = operating_expenses * 0.6 + annual_debt_service
    break_even_revenue = round(fixed_annual_costs / gross_margin_ratio, 2) if gross_margin_ratio > 0 else 0

    # 6. Dynamic Tax Calculations
    # GST Composition Scheme: 1% for traders, 2% for manufacturers, exempt for raw agri
    if sk in ("retail", "trading", "shop"):
        gst_rate = 0.01
        gst_regime_name = "GST Composition Scheme (1% flat turnover tax for traders)"
    elif sk in ("dairy", "agriculture"):
        gst_rate = 0.00
        gst_regime_name = "GST Nil / Exempt (Raw dairy & agricultural produce exempt under GST)"
    else:
        gst_rate = 0.02
        gst_regime_name = "GST Composition Scheme (2% flat turnover tax for small manufacturers)"

    estimated_annual_gst = round(projected_annual_revenue * gst_rate, 2)

    # Income Tax: Presumptive Taxation u/s 44AD (6% on digital, 8% on cash)
    lit_band = market_profile.get("literacy_band", "Moderate")
    digital_share = 0.70 if lit_band == "High" else (0.45 if lit_band == "Moderate" else 0.20)
    blended_presumptive_rate = (digital_share * 0.06) + ((1 - digital_share) * 0.08)
    
    presumptive_income = round(projected_annual_revenue * blended_presumptive_rate, 2)
    # Income Tax calculation (New Tax Regime for Individuals / Sole Proprietorship: 0-3L Nil, 3-7L 5% with rebate u/s 87A)
    # Under section 87A, income up to 7L has zero effective tax under new regime.
    if presumptive_income <= 700_000:
        estimated_income_tax = 0.0
    else:
        # standard 10% slab on excess over 7L
        estimated_income_tax = round((presumptive_income - 700_000) * 0.10, 2)

    income_tax_regime = f"Presumptive Taxation u/s 44AD (Estimated {round(blended_presumptive_rate*100, 1)}% taxable profit)"
    total_annual_tax = round(estimated_annual_gst + estimated_income_tax, 2)
    net_profit_after_tax = round(net_profit_before_tax - total_annual_tax, 2)
    return_on_margin_pct = round((net_profit_after_tax / margin_capital) * 100, 1)

    return FinancialStatement(
        margin_capital=margin_capital,
        project_cost=project_cost,
        loan_amount=gross_loan,
        subsidy_amount=subsidy_amount,
        effective_loan=effective_loan,
        scheme_name=scheme_name,
        interest_rate_annual=annual_rate,
        tenure_years=tenure_years,
        moratorium_months=moratorium_months,
        quarterly_emi=quarterly_emi,
        total_interest_payable=total_interest,
        total_repayment=total_repayment,
        projected_annual_revenue=projected_annual_revenue,
        cogs_amount=cogs_amount,
        gross_profit=gross_profit,
        gross_margin_pct=gross_margin_pct,
        operating_expenses=operating_expenses,
        ebitda=ebitda,
        annual_debt_service=annual_debt_service,
        net_profit_before_tax=net_profit_before_tax,
        dscr=dscr,
        break_even_revenue=break_even_revenue,
        gst_regime=gst_regime_name,
        estimated_annual_gst=estimated_annual_gst,
        income_tax_regime=income_tax_regime,
        presumptive_income=presumptive_income,
        estimated_annual_income_tax=estimated_income_tax,
        total_annual_tax=total_annual_tax,
        net_profit_after_tax=net_profit_after_tax,
        return_on_margin_pct=return_on_margin_pct,
    )


if __name__ == "__main__":
    test_profile = {
        "location": "Sudasana (VILLAGE), Mahesana, Gujarat",
        "location_type": "Rural Village",
        "dominant_economy": "Cottage & Artisan Handloom Cluster",
        "consumer_base_size": "Medium",
        "literacy_band": "High",
        "is_rural": True,
        "sc_st_ratio_pct": 13.6,
    }
    stmt = build_financial_plan(margin_capital=100_000, sector="textiles", market_profile=test_profile)
    print("Project Cost:", stmt.project_cost)
    print("Scheme:", stmt.scheme_name, "@", stmt.interest_rate_annual, "%")
    print("Quarterly EMI:", stmt.quarterly_emi)
    print("Annual Revenue:", stmt.projected_annual_revenue)
    print("Total Tax:", stmt.total_annual_tax)
    print("Net Profit After Tax:", stmt.net_profit_after_tax)
