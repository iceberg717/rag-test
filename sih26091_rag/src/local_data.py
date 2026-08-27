"""
Hyper-local structured data lookup — backed by the real Mahesana
district Census dataset (Mahesana_Population.csv).

PURPOSE IN THIS PROJECT:
  The CSV is NOT used to show census statistics to the user.
  It is used to silently profile the economic character of the user's
  location so that the tax estimates, interest rate, and investment advice
  given to a small-enterprise owner are dynamically calibrated to actual
  local conditions (e.g. an agriculture-dominant village gets different
  interest rate subsidies, GST treatment, and investment risk advice than
  a commercial trading town or artisan cluster).
"""

import csv
import os

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "Mahesana_Population.csv"
)


def load_local_stats():
    """Return all rows from the Mahesana population CSV as a list of dicts."""
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def find_village(name: str):
    """
    Look up a village, town, or sub-district by name (case-insensitive).
    Prioritizes specific village/town rows over aggregate sub-district rows.
    """
    rows = load_local_stats()
    name_clean = name.strip().lower()

    # Pass 1: Exact match on VILLAGE or TOWN level
    for row in rows:
        r_name = row["Name"].strip().lower()
        if r_name == name_clean and row["Level"] in ("VILLAGE", "TOWN"):
            return _normalise(row)

    # Pass 2: Cleaned exact match ignoring suffix like '(Rural)' or '(M)'
    for row in rows:
        r_name = row["Name"].strip().lower().split("(")[0].strip()
        if r_name == name_clean and row["Level"] in ("VILLAGE", "TOWN"):
            return _normalise(row)

    # Pass 3: Exact match on any level (SUB-DISTRICT, DISTRICT)
    for row in rows:
        if row["Name"].strip().lower() == name_clean:
            return _normalise(row)

    # Pass 4: Substring match on VILLAGE / TOWN
    for row in rows:
        r_name = row["Name"].strip().lower()
        if name_clean in r_name and row["Level"] in ("VILLAGE", "TOWN"):
            return _normalise(row)

    # Pass 5: Substring match anywhere
    for row in rows:
        if name_clean in row["Name"].strip().lower():
            return _normalise(row)

    return None


def _normalise(row: dict) -> dict:
    """Convert raw CSV row into a structured demographic & economic profile."""
    tot_p  = _safe_int(row["TOT_P"])
    lit    = _safe_int(row["P_LIT"])
    work   = _safe_int(row["TOT_WORK_P"])
    hh     = _safe_int(row["No_HH"])
    p06    = _safe_int(row["P_06"])
    sc_p   = _safe_int(row["P_SC"])
    st_p   = _safe_int(row["P_ST"])
    cult   = _safe_int(row["MAIN_CL_P"])
    agri_l = _safe_int(row["MAIN_AL_P"])
    hhi    = _safe_int(row["MAIN_HH_P"])
    ot     = _safe_int(row["MAIN_OT_P"])
    non_w  = _safe_int(row["NON_WORK_P"])

    literacy_rate  = round((lit / tot_p * 100), 1) if tot_p else 0
    work_part_rate = round((work / tot_p * 100), 1) if tot_p else 0
    sc_st_rate     = round(((sc_p + st_p) / tot_p * 100), 1) if tot_p else 0

    return {
        "village"                : row["Name"],
        "level"                  : row["Level"],
        "tru"                    : row["TRU"],
        "district"               : "Mahesana",
        "state"                  : "Gujarat",
        "population"             : tot_p,
        "households"             : hh,
        "literate"               : lit,
        "literacy_rate_pct"      : literacy_rate,
        "total_workers"          : work,
        "cultivators"            : cult,
        "agricultural_labourers" : agri_l,
        "household_industry"     : hhi,
        "other_workers"          : ot,
        "non_workers"            : non_w,
        "sc_population"          : sc_p,
        "st_population"          : st_p,
        "sc_st_ratio_pct"        : sc_st_rate,
        "work_participation_pct" : work_part_rate,
    }


def derive_market_profile(village_row: dict, sector: str) -> dict:
    """
    Translate census data into FINANCIAL & CREDIT SIGNALS for small enterprise
    tax prediction, interest rate estimation, and investment planning.
    """
    if not village_row:
        return {
            "location"              : "Generic Mahesana Region, Gujarat",
            "location_type"         : "semi-urban hub",
            "dominant_economy"      : "mixed commerce",
            "consumer_base_size"    : "medium",
            "literacy_band"         : "moderate",
            "priority_lending_zone" : False,
            "special_subsidy_tier"  : "Standard General Tier (15-25% PMEGP)",
            "applicable_tax_regime" : "GST Composition Scheme (1% trade / 2% mfg)",
            "market_saturation"     : "moderate",
            "sector_demand_signal"  : "MODERATE",
            "investment_risk_note"  : "Standard MSME risk",
        }

    pop     = village_row.get("population", 0)
    hh      = village_row.get("households", 0)
    lit     = village_row.get("literacy_rate_pct", 0)
    work    = village_row.get("total_workers", 0) or 1
    cult    = village_row.get("cultivators", 0)
    agri_l  = village_row.get("agricultural_labourers", 0)
    hhi     = village_row.get("household_industry", 0)
    ot      = village_row.get("other_workers", 0)
    level   = village_row.get("level", "VILLAGE")
    tru     = village_row.get("tru", "Rural")
    sc_st_r = village_row.get("sc_st_ratio_pct", 0)

    # --- Location type ---
    if tru == "Urban" or level == "TOWN":
        location_type = "Urban / Semi-Urban Commercial Town"
        is_rural = False
    elif pop > 4000:
        location_type = "Large Rural Village Hub"
        is_rural = True
    else:
        location_type = "Rural Village"
        is_rural = True

    # --- Dominant economy ---
    agri_share  = (cult + agri_l) / work
    hhi_share   = hhi / work
    trade_share = ot / work

    if agri_share >= 0.45:
        dominant_economy = "Agrarian & Dairy Ecosystem"
    elif hhi_share >= 0.10:
        dominant_economy = "Cottage & Artisan Handloom Cluster"
    elif trade_share >= 0.35:
        dominant_economy = "Commercial Trade & Services"
    else:
        dominant_economy = "Diversified Rural Economy"

    # --- Consumer base & Market scale ---
    if hh >= 1500:
        consumer_base_size = "Large"
        estimated_market_turnover_capacity = "High (Rs.15L - Rs.50L+ potential)"
    elif hh >= 400:
        consumer_base_size = "Medium"
        estimated_market_turnover_capacity = "Moderate (Rs.8L - Rs.25L potential)"
    else:
        consumer_base_size = "Niche / Micro"
        estimated_market_turnover_capacity = "Local / Concentrated (Rs.3L - Rs.12L potential)"

    # --- Literacy & Fintech / Digital Tax compliance readiness ---
    if lit >= 78:
        literacy_band = "High"
        tax_compliance_mode = "Direct Online GST / QR-based UPI (6% presumptive tax u/s 44AD)"
    elif lit >= 60:
        literacy_band = "Moderate"
        tax_compliance_mode = "CSC / Jan Seva Kendra / CA assisted filing"
    else:
        literacy_band = "Low"
        tax_compliance_mode = "Presumptive Tax u/s 44AD (simplified cash accounting)"

    # --- Priority Lending & Subsidy Tier ---
    # Rural areas qualify for 35% PMEGP subsidy vs 25% urban. High SC/ST qualifies for 35% + special interest subvention
    if is_rural and sc_st_r >= 15.0:
        special_subsidy_tier = "Priority Rural & Special Category Tier (Up to 35% PMEGP capital subsidy + 3% interest subvention)"
        priority_zone = True
    elif is_rural:
        special_subsidy_tier = "Rural Enterprise Tier (25% - 35% PMEGP subsidy / NABARD priority rate)"
        priority_zone = True
    else:
        special_subsidy_tier = "Urban MSME General Tier (15% - 25% subsidy under MUDRA / PMEGP)"
        priority_zone = False

    # --- Sector alignment & dynamic risk assessment ---
    sk = sector.strip().lower()
    if sk in ("dairy", "agriculture", "agri", "farming", "cattle"):
        if agri_share >= 0.35:
            sector_demand = "HIGH — strong local fodder/agri supply chain & steady cash flow"
            saturation = "Moderate (established cooperative network)"
            risk_note = "Low default risk; qualified for NABARD / KCC interest concession"
        else:
            sector_demand = "MODERATE — consumer demand present, supply must be sourced"
            saturation = "Low (under-served)"
            risk_note = "Moderate risk; focus on cold chain and hygiene"
    elif sk in ("textile", "textiles", "weaving", "handicraft", "garment", "embroidery"):
        if hhi_share >= 0.05 or dominant_economy == "Cottage & Artisan Handloom Cluster":
            sector_demand = "VERY HIGH — native artisan skills available, export & regional link viable"
            saturation = "Moderate (clustering advantage)"
            risk_note = "Low-Medium risk; eligible for Gujarat Khadi & Village Industries board incentives"
        else:
            sector_demand = "MODERATE — local tailoring & festive demand"
            saturation = "Low"
            risk_note = "Moderate risk; inventory turnaround critical"
    elif sk in ("retail", "trading", "shop", "grocery", "kirana"):
        if trade_share >= 0.25:
            sector_demand = "HIGH — strong purchasing footfall"
            saturation = "High (differentiation required)"
            risk_note = "Medium risk; margin competition; digital billing recommended"
        else:
            sector_demand = "MODERATE to HIGH — essential goods demand"
            saturation = "Low to Moderate"
            risk_note = "Low-Medium risk; steady daily turnover"
    else:
        sector_demand = "MODERATE — emergent sector"
        saturation = "Low (first-mover advantage)"
        risk_note = "Medium risk; requires local awareness"

    # --- Tax Regime Note ---
    tax_note = (
        "Eligible for GST Composition Scheme (1% flat for traders, 2% for manufacturers) "
        "if annual turnover is under Rs.1.5 Crore. Alternatively, Presumptive Taxation u/s 44AD "
        "allows declaring 6% of digital turnover or 8% of non-digital turnover as taxable income."
    )

    return {
        "location"                           : f"{village_row.get('village')} ({village_row.get('level')}), Mahesana, Gujarat",
        "location_type"                      : location_type,
        "is_rural"                           : is_rural,
        "dominant_economy"                   : dominant_economy,
        "consumer_base_size"                 : consumer_base_size,
        "market_turnover_capacity"           : estimated_market_turnover_capacity,
        "literacy_band"                      : literacy_band,
        "tax_compliance_mode"                : tax_compliance_mode,
        "priority_lending_zone"              : priority_zone,
        "special_subsidy_tier"               : special_subsidy_tier,
        "sc_st_ratio_pct"                    : sc_st_r,
        "applicable_tax_regime"              : tax_note,
        "market_saturation"                  : saturation,
        "sector_demand_signal"               : sector_demand,
        "investment_risk_note"               : risk_note,
    }


if __name__ == "__main__":
    for place in ["Sudasana", "Visnagar", "Chelana"]:
        row = find_village(place)
        profile = derive_market_profile(row, "textiles")
        print(f"\n=== {place} Profile ===")
        for k, v in profile.items():
            print(f"  {k:30s}: {v}")
