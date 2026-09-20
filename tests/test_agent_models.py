from contextlab.agent.models import AgentState, EventKind, Message


def test_canonical_state_preserves_messages_and_events() -> None:
    state = AgentState(
        experiment_id="e1",
        run_id="r1",
        task_id="t1",
        policy_id="full-history",
        system_instruction="safe",
        task_instruction="fix it",
        objective="tests pass",
    )
    state.canonical_messages.append(Message(role="user", content="inspect"))
    event = state.emit(EventKind.STATUS, status="running")
    assert event.run_id == "r1"
    assert state.canonical_messages[0].content == "inspect"
