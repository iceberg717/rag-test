"""
Module 1 support: RAG engine for the Hyper-Local Business Feasibility Report.

Retrieval: TF-IDF + cosine similarity over chunked knowledge base docs.
  (Swap in sentence-transformers/FAISS or an embeddings API later for
  semantic retrieval — TF-IDF is used here so the whole pipeline runs
  instantly with zero model downloads, good for a hackathon demo.)

Generation: Google Gemini (reuses the same google-genai client pattern
  you were already testing). Falls back to a template-based report if
  no API key is available, so the pipeline still runs end-to-end offline.
"""

import os
import csv
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dummy_docs")
CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "Mahesana_Population.csv")


def load_and_chunk_documents(docs_dir: str = DOCS_DIR, chunk_marker: str = "[DOCUMENT:"):
    """
    Splits each .txt file on the [DOCUMENT: ...] marker so each chunk is one
    self-contained knowledge unit (scheme rule, sector note, etc).
    """
    chunks = []
    for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        raw_chunks = content.split(chunk_marker)
        for c in raw_chunks:
            c = c.strip()
            if not c:
                continue
            chunks.append(chunk_marker + c if not c.startswith("DOCUMENT") else c)
    return chunks


class SimpleRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def load_and_chunk_csv(csv_path: str = CSV_PATH, tru_filter: str = "Total") -> list:
    """
    Converts each VILLAGE / TOWN row from Mahesana_Population.csv into a
    concise economic-character description for TF-IDF retrieval.

    PURPOSE: When a user's location is queried, the retriever can surface
    descriptions of similar localities (same economic type, same sector
    workforce profile) to ground the tax/investment advisory in real,
    comparable local examples.

    Only rows with non-zero population are included.
    """
    def _safe_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    chunks = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Level"] not in ("VILLAGE", "TOWN"):
                    continue
                if row["TRU"].strip() != tru_filter:
                    continue

                tot_p   = _safe_int(row["TOT_P"])
                if tot_p == 0:
                    continue

                hh      = _safe_int(row["No_HH"])
                lit     = _safe_int(row["P_LIT"])
                work    = _safe_int(row["TOT_WORK_P"]) or 1
                main_cl = _safe_int(row["MAIN_CL_P"])
                main_al = _safe_int(row["MAIN_AL_P"])
                main_hh = _safe_int(row["MAIN_HH_P"])
                main_ot = _safe_int(row["MAIN_OT_P"])

                lit_rate   = round(lit / tot_p * 100, 1) if tot_p else 0
                agri_share = round((main_cl + main_al) / work * 100, 1)
                hhi_share  = round(main_hh / work * 100, 1)
                trade_share= round(main_ot / work * 100, 1)

                # Describe economic character — useful for advisory retrieval
                if agri_share >= 50:
                    econ_type = "agriculture-dominant"
                elif hhi_share >= 20:
                    econ_type = "household-industry oriented"
                elif trade_share >= 40:
                    econ_type = "trade and services oriented"
                else:
                    econ_type = "mixed economy"

                hh_size = "large" if hh >= 1000 else "medium" if hh >= 300 else "small"
                lit_band = "high" if lit_rate >= 75 else "moderate" if lit_rate >= 55 else "low"

                chunk = (
                    f"{row['Name']} is a {row['Level'].lower()} in Mahesana district, Gujarat. "
                    f"It has a {econ_type} local economy with a {hh_size} consumer base. "
                    f"Workforce composition: {agri_share}% in agriculture, "
                    f"{hhi_share}% in household industry, {trade_share}% in trade/other. "
                    f"Literacy rate: {lit_rate}% ({lit_band} literacy). "
                    f"This profile affects GST filing capacity, digital payment adoption, "
                    f"investment risk, and local demand for small enterprise products. "
                    f"Suitable sectors: "
                    f"{'dairy/agro-processing' if agri_share >= 40 else ''} "
                    f"{'textile/handicraft' if hhi_share >= 15 else ''} "
                    f"{'retail/trading' if trade_share >= 30 else 'mixed services'}."
                )
                chunks.append(chunk)
    except FileNotFoundError:
        pass
    return chunks


def build_retriever() -> SimpleRetriever:
    doc_chunks = load_and_chunk_documents()
    csv_chunks = load_and_chunk_csv()
    all_chunks = doc_chunks + csv_chunks
    if not all_chunks:
        raise RuntimeError("No chunks found — check docs dir and CSV path.")
    return SimpleRetriever(all_chunks)


def generate_with_gemini(prompt: str) -> str:
    """Calls Gemini if GEMINI_API_KEY is set; raises if not configured."""
    # pyrefly: ignore [missing-import]
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model="gemini-3.6-flash")
    response = chat.send_message(prompt)
    return response.text


def generate_advisory_report(plan, market_profile: dict,
                              location: str, sector: str,
                              retrieved_chunks: list) -> str:
    """
    Builds a location-aware Tax & Investment Advisory Report using Gemini.
    """
    context_text = "\n\n".join(chunk for chunk, score in retrieved_chunks)

    # Summarise the financial statement for the prompt
    fs_text = f"""- Project Cost: Rs.{plan.project_cost:,.2f} (Margin Money: Rs.{plan.margin_capital:,.2f})
- Loan Amount: Rs.{plan.loan_amount:,.2f} | Scheme: {plan.scheme_name}
- Area-Adjusted Interest Rate: {plan.interest_rate_annual}% p.a. | Tenure: {plan.tenure_years} yrs (Moratorium: {plan.moratorium_months} mos)
- Subsidy Eligibility: Rs.{plan.subsidy_amount:,.2f} | Effective Loan Burden: Rs.{plan.effective_loan:,.2f}
- Quarterly EMI: Rs.{plan.quarterly_emi:,.2f} | Annual Debt Service: Rs.{plan.annual_debt_service:,.2f}
- Projected Annual Revenue: Rs.{plan.projected_annual_revenue:,.2f}
- Direct Costs (COGS): Rs.{plan.cogs_amount:,.2f} | Gross Margin: {plan.gross_margin_pct}% (Rs.{plan.gross_profit:,.2f})
- Operating Expenses: Rs.{plan.operating_expenses:,.2f} | EBITDA: Rs.{plan.ebitda:,.2f}
- Net Profit Before Tax: Rs.{plan.net_profit_before_tax:,.2f} | DSCR: {plan.dscr}x
- Break-Even Annual Turnover: Rs.{plan.break_even_revenue:,.2f}
- Estimated Annual GST: Rs.{plan.estimated_annual_gst:,.2f} ({plan.gst_regime})
- Presumptive Income (u/s 44AD): Rs.{plan.presumptive_income:,.2f} | Estimated Income Tax: Rs.{plan.estimated_annual_income_tax:,.2f}
- Total Annual Tax: Rs.{plan.total_annual_tax:,.2f} | Net Profit After Tax: Rs.{plan.net_profit_after_tax:,.2f}
- Projected Return on Margin: {plan.return_on_margin_pct}%"""

    mp_text = "\n".join(f"  {k}: {v}" for k, v in market_profile.items()) if market_profile else \
              "  (Using generic Mahesana district profile)"

    prompt = f"""You are an expert small enterprise tax advisor and financial planner in Gujarat, India.
A rural/semi-urban micro-entrepreneur is setting up a {sector} enterprise at {location}, Mahesana District, Gujarat.

LOCATION PROFILE (derived from Mahesana Census & Economic Data):
{mp_text}

PROJECTED FINANCIAL STATEMENT & TAX ESTIMATES (Baseline Calculations):
{fs_text}

KNOWLEDGE BASE & SECTOR GUIDELINES:
{context_text}

Based on the location's purchasing power, workforce composition, prevailing interest rate ({plan.interest_rate_annual}%), and scheme benefits, generate a comprehensive, professional **Tax & Investment Advisory Report** with the following sections:

1. FINANCIAL STATEMENT ANALYSIS & REVENUE FEASIBILITY
   - Review of projected annual revenue (Rs.{plan.projected_annual_revenue:,.0f}) against local purchasing power and consumer base.
   - Operating margin viability, raw material sourcing, and Debt-Service Coverage Ratio ({plan.dscr}x).
   - Monthly break-even revenue requirement and safe operating thresholds.

2. COMPREHENSIVE TAX ESTIMATION & OPTIMIZATION
   - GST Strategy: Recommend GST Composition Scheme vs Regular GST for this {sector} business in this area.
   - Presumptive Taxation u/s 44AD / 44ADA: Explain why declaring presumptive profit ({round(plan.presumptive_income, 0)}) saves compliance costs.
   - Estimated Annual Tax Breakdown: GST (Rs.{plan.estimated_annual_gst:,.0f}) + Income Tax (Rs.{plan.estimated_annual_income_tax:,.0f}) = Total Rs.{plan.total_annual_tax:,.0f}.
   - Tax saving and deduction opportunities (Section 80C, MSME depreciation, Input Tax Credit).

3. CAPITAL ALLOCATION & INVESTMENT BLUEPRINT
   - Detailed recommended allocation of the Rs.{plan.project_cost:,.0f} total capital:
     * Fixed Assets & Machinery (tools, equipment, workspace setup)
     * Working Capital & Raw Materials (initial stock buffer)
     * Marketing & Local Market Outreach (signage, community engagement)
     * Contingency & EMI Emergency Reserve (3-6 months buffer)
   - Specific equipment or assets recommended for a {sector} enterprise in {location}.

4. LENDING RATE & GOVERNMENT SUBSIDY ADVISORY
   - Explanation of why the {plan.interest_rate_annual}% interest rate applies under {plan.scheme_name}.
   - How to claim the Rs.{plan.subsidy_amount:,.0f} government subsidy through Gujarat State Portal / DIC Mahesana / PMEGP e-Portal.

5. LOCAL STRATEGIC & RISK RECOMMENDATIONS
   - Area-specific advantages (workforce availability, local demand).
   - Recommended payment/billing method (UPI, digital ledger, invoice management).
   - Practical steps to start operations within 30 days.

Make the output well-formatted with markdown tables, clear bullet points, and actionable financial advice."""

    try:
        return generate_with_gemini(prompt)
    except Exception as e:
        return _fallback_advisory_report(plan, market_profile, location, sector, context_text, reason=str(e))


def _fallback_advisory_report(plan, market_profile, location, sector, context_text, reason=""):
    """Dynamic fallback report when Gemini API is unavailable."""
    return f"""[OFFLINE FINANCIAL ADVISORY MODE — Gemini status: {reason}]

╔══════════════════════════════════════════════════════════════════════════════╗
   TAX, INVESTMENT & FINANCIAL STATEMENT ADVISORY REPORT
   Enterprise: {sector.title()} | Location: {location}
╚══════════════════════════════════════════════════════════════════════════════╝

1. PROJECTED FINANCIAL STATEMENT (P&L)
   ─────────────────────────────────────────────────────────────────────────────
   Projected Annual Revenue : Rs.{plan.projected_annual_revenue:,.2f}
   Cost of Goods Sold (COGS): Rs.{plan.cogs_amount:,.2f}
   Gross Profit             : Rs.{plan.gross_profit:,.2f} ({plan.gross_margin_pct}% margin)
   Operating Expenses (Opex): Rs.{plan.operating_expenses:,.2f}
   EBITDA (Operating Profit): Rs.{plan.ebitda:,.2f}
   Annual Debt Service (EMI): Rs.{plan.annual_debt_service:,.2f}
   Net Profit (Pre-Tax)     : Rs.{plan.net_profit_before_tax:,.2f}
   Debt-Service Coverage    : {plan.dscr}x  ({'Healthy' if plan.dscr >= 1.5 else 'Tight margin'})
   Annual Break-Even Revenue: Rs.{plan.break_even_revenue:,.2f}
   ─────────────────────────────────────────────────────────────────────────────

2. ESTIMATED ANNUAL TAX LIABILITY
   ─────────────────────────────────────────────────────────────────────────────
   GST Regime Applicable    : {plan.gst_regime}
   Estimated Annual GST     : Rs.{plan.estimated_annual_gst:,.2f}
   Income Tax Regime        : {plan.income_tax_regime}
   Presumptive Income       : Rs.{plan.presumptive_income:,.2f}
   Estimated Income Tax     : Rs.{plan.estimated_annual_income_tax:,.2f}
   TOTAL ANNUAL TAX BURDEN  : Rs.{plan.total_annual_tax:,.2f}
   ─────────────────────────────────────────────────────────────────────────────
   Net Profit After Tax     : Rs.{plan.net_profit_after_tax:,.2f}
   Return on Margin Capital : {plan.return_on_margin_pct}% p.a.

3. LOAN & AREA INTEREST RATE PROFILE
   ─────────────────────────────────────────────────────────────────────────────
   Scheme Name              : {plan.scheme_name}
   Prevailing Interest Rate : {plan.interest_rate_annual}% p.a.
   Project Cost / Loan      : Rs.{plan.project_cost:,.2f} / Rs.{plan.loan_amount:,.2f}
   Capital Subsidy Eligible : Rs.{plan.subsidy_amount:,.2f}
   Effective Net Loan       : Rs.{plan.effective_loan:,.2f}
   Quarterly EMI            : Rs.{plan.quarterly_emi:,.2f} (Tenure: {plan.tenure_years} yrs, Moratorium: {plan.moratorium_months} mos)

4. RECOMMENDED CAPITAL DEPLOYMENT (Rs.{plan.project_cost:,.0f})
   - Fixed Assets & Equipment (50%) : Rs.{plan.project_cost * 0.50:,.0f}
   - Working Capital & Stock  (30%) : Rs.{plan.project_cost * 0.30:,.0f}
   - Marketing & Setup        (10%) : Rs.{plan.project_cost * 0.10:,.0f}
   - Contingency / Reserve    (10%) : Rs.{plan.project_cost * 0.10:,.0f}

5. LOCATION PROFILE & COMPLIANCE NOTES
   - Location Type          : {market_profile.get('location_type', 'Rural Village')}
   - Dominant Economy       : {market_profile.get('dominant_economy', 'Agrarian')}
   - Sector Demand Signal   : {market_profile.get('sector_demand_signal', 'MODERATE')}
   - Recommended Tax Filing : {market_profile.get('tax_compliance_mode', 'Online GST / Presumptive')}
"""


# Alias for backward compatibility
generate_feasibility_report = generate_advisory_report
_fallback_template_report = _fallback_advisory_report



if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.retrieve("dairy sector opportunities and threats", top_k=3)
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk[:80]}...")
