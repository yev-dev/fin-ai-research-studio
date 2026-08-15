from dashboard.financial_analyst_app import format_agent_prompt, preview_filename


def test_format_agent_prompt_with_template():
    out = format_agent_prompt("Single Asset Analysis", asset="AAPL")
    assert "Analyze the asset AAPL" in out


def test_format_agent_prompt_missing_template():
    out = format_agent_prompt("NoSuchTemplate", asset="XYZ")
    assert out == "NoSuchTemplate"


def test_preview_filename_basic():
    name = preview_filename("My Report: $ Test", agent_name="agent", ext="pdf")
    assert name.endswith(".pdf")
    assert "My_Report" in name
