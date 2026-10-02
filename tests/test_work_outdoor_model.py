from __future__ import annotations

import numpy as np
import pandas as pd

from work_outdoor_model import (
    WorkOutdoorConfig,
    compute_work_outdoor_probs,
    normalize_soc_code,
    parse_naics_group,
)


def test_parse_naics_group_handles_common_das_values():
    assert parse_naics_group("naics62_health_care")[0] == "62"
    assert parse_naics_group("NAICS31_33_manufacturing")[0] == "31_33"
    assert parse_naics_group("not_working") == (None, "not_working")
    assert parse_naics_group(None) == (None, "missing")
    assert parse_naics_group("unknown industry") == (None, "unparsed")


def test_normalize_soc_code_keeps_detailed_soc_codes_only():
    assert normalize_soc_code("11-1011.00") == "11-1011"
    assert normalize_soc_code("11-1011") == "11-1011"
    assert normalize_soc_code("11-0000") == "11-0000"
    assert normalize_soc_code("not-a-soc") is None


def test_compute_work_outdoor_probs_uses_oews_weights_and_onet_context(tmp_path):
    oews_path = tmp_path / "oews.csv"
    onet_path = tmp_path / "onet.csv"

    oews_path.write_text(
        "\n".join(
            [
                "NAICS,OCC_CODE,O_GROUP,TOT_EMP",
                "110000,11-1011,detailed,100",
                "110000,13-2011,detailed,300",
            ]
        ),
        encoding="utf-8",
    )
    onet_path.write_text(
        "\n".join(
            [
                "O*NET-SOC Code,Element Name,Scale ID,Category,Data Value",
                "11-1011.00,\"Outdoors, Exposed to All Weather Conditions\",CXP,4,50",
                "11-1011.00,\"Outdoors, Exposed to All Weather Conditions\",CXP,5,30",
                "11-1011.00,\"Outdoors, Under Cover\",CXP,4,10",
                "13-2011.00,\"Outdoors, Exposed to All Weather Conditions\",CXP,4,10",
                "13-2011.00,\"Outdoors, Under Cover\",CXP,4,20",
            ]
        ),
        encoding="utf-8",
    )

    people = pd.DataFrame(
        {
            "person_id": ["p1", "p2"],
            "trip_taker_industry": ["naics11_agriculture", "not_working"],
        }
    )
    cfg = WorkOutdoorConfig(
        oews_nat4d_path=oews_path,
        onet_work_context_path=onet_path,
        outdoor_elements=[
            "Outdoors, Exposed to All Weather Conditions",
            "Outdoors, Under Cover",
        ],
        outdoor_category_threshold=4,
        combine_elements="union",
        oews_occ_group="detailed",
    )

    person_probs, sector_probs, missingness = compute_work_outdoor_probs(people, cfg)

    # SOC outdoor probabilities are 0.82 and 0.28, weighted by 100 and 300 workers.
    expected_sector_probability = (0.82 * 100 + 0.28 * 300) / 400
    assert np.isclose(sector_probs.loc[sector_probs["sector"] == "11", "p_outdoor_work"].iloc[0], expected_sector_probability)
    assert np.isclose(person_probs.loc[person_probs["person_id"] == "p1", "p_outdoor_work"].iloc[0], expected_sector_probability)
    assert person_probs.loc[person_probs["person_id"] == "p2", "p_outdoor_reason"].iloc[0] == "not_working"
    assert missingness["step"].str.contains("person_assign").any()
