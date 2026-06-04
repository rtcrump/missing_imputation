"""FACT-E item definitions and longitudinal visit structure.

This module encodes the *structure* of the FACT-E (Functional Assessment of
Cancer Therapy - Esophageal) questionnaire used by the study: which item codes
belong to which subscale, and the ordered list of REDCap study visits. This is
public instrument metadata, not patient data.

The 44 FACT-E items are 0-4 Likert-scored. Subscales:
  - PWB  (Physical Well-Being):      gp1..gp7
  - SWB  (Social/Family Well-Being): gs1..gs7
  - EWB  (Emotional Well-Being):     ge1..ge6
  - FWB  (Functional Well-Being):    gf1..gf7
  - ECS  (Esophageal Cancer Subscale): a_hn1..a_hn5, a_hn7, a_hn10,
                                        a_e1..a_e7, a_c6, a_c2, a_act11
"""

# Subscale -> item codes
SUBSCALES = {
    "PWB": [f"gp{i}" for i in range(1, 8)],
    "SWB": [f"gs{i}" for i in range(1, 8)],
    "EWB": [f"ge{i}" for i in range(1, 7)],
    "FWB": [f"gf{i}" for i in range(1, 8)],
    "ECS": (
        [f"a_hn{i}" for i in range(1, 6)] + ["a_hn7", "a_hn10"]
        + [f"a_e{i}" for i in range(1, 8)] + ["a_c6", "a_c2", "a_act11"]
    ),
}

# Flat list of all 44 FACT-E item columns, in canonical order.
FACTE_COLUMNS = [col for items in SUBSCALES.values() for col in items]

# Valid Likert response range for every FACT-E item.
LIKERT_MIN = 0
LIKERT_MAX = 4

# Ordered REDCap longitudinal study visits (event names), earliest to latest.
VISIT_ORDER = {
    "baseline_arm_1": 0,
    "preoperative_arm_1": 1,
    "1_month_postop_arm_1": 2,
    "3_months_postop_arm_1": 3,
    "6_months_postop_arm_1": 4,
    "1_year_postop_arm_1": 5,
    "2_years_postop_arm_1": 6,
    "3_years_postop_arm_1": 7,
    "4_years_postop_arm_1": 8,
    "5_years_postop_arm_1": 9,
}

# Visit event names in chronological order.
VISITS = sorted(VISIT_ORDER, key=VISIT_ORDER.get)

__all__ = [
    "SUBSCALES",
    "FACTE_COLUMNS",
    "LIKERT_MIN",
    "LIKERT_MAX",
    "VISIT_ORDER",
    "VISITS",
]
