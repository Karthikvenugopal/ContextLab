import pytest

from contextlab.agent.protocol import MalformedDecision, parse_decision


def test_parses_valid_tool_action() -> None:
    result = parse_decision(
        '```json\n{"action":"tool","tool":{"name":"read_file","arguments":{"path":"a.py"}}}\n```'
    )
    assert result.tool and result.tool.name == "read_file"


def test_rejects_unstructured_model_output() -> None:
    with pytest.raises(MalformedDecision):
        parse_decision("I think we should inspect the project")
