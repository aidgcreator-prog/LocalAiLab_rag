from streamlit_sidebar import PIPELINE_PRESET_STEPS
from streamlit_orchestration import get_pipeline_step_agents


def test_sidebar_pipeline_presets_match_expected_agent_flows():
    assert get_pipeline_step_agents(PIPELINE_PRESET_STEPS["rag_presentation"]) == ["ragsub", "presenter"]
    assert get_pipeline_step_agents(PIPELINE_PRESET_STEPS["data_presentation"]) == ["data_scientist", "presenter"]
    assert get_pipeline_step_agents(PIPELINE_PRESET_STEPS["search_write"]) == ["websearch", "writer"]
