"""Read-only readiness view for AlphaForge's local and licensed data paths."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from api.service import DISCLAIMER, data_connections_status
from app.ui import footer, page


page("Data connections", "Research-data readiness / Provenance checks", section="Data connections")
status = data_connections_status()
bundle = status["licensed_bundle"]

st.subheader("Licensed research bundle")
one, two, three, four = st.columns(4)
one.metric("Connection", "Ready" if bundle["ready_for_price_research"] else "Not ready")
two.metric("Point-in-time fundamentals", "Verified" if bundle["point_in_time_fundamentals"] else "Not attested")
three.metric("Historical universe", "Verified" if bundle["survivorship_free_universe"] else "Not attested")
four.metric("Delisted companies", "Included" if bundle["includes_delisted_securities"] else "Not attested")

if bundle["provider_name"]:
    st.caption(f"Declared source: {bundle['provider_name']}")
if bundle["ready_for_price_research"]:
    st.success("Licensed price research is available in Strategy Lab and Portfolio Lab.", icon=":material/check_circle:")
else:
    st.info("No licensed bundle is active. Free and synthetic research paths remain available.", icon=":material/info:")
for issue in bundle["issues"]:
    st.warning(issue, icon=":material/warning:")

st.subheader("Fundamental data adapters")
adapters = pd.DataFrame(status["fundamental_adapters"])
adapters["configured"] = adapters["configured"].map({True: "Connected", False: "Not connected"})
adapters["point_in_time_fundamentals"] = adapters["point_in_time_fundamentals"].map(
    {True: "Filing-dated", False: "Restated / blocked"}
)
st.dataframe(
    adapters.rename(columns={
        "label": "Source", "configured": "Connection", "point_in_time_fundamentals": "Fundamental timing",
        "note": "Research status",
    })[["Source", "Connection", "Fundamental timing", "Research status"]],
    use_container_width=True,
    hide_index=True,
)

st.info("AlphaForge reads connection status only. Keys, licensed files, and raw vendor data remain local.", icon=":material/lock:")
footer(DISCLAIMER)
