from app.browser.action_executor import ActionExecutor


class _DummySession:
    page = None


def test_scale_coordinates_uses_1000x1000_reference_and_clamps():
    executor = ActionExecutor(_DummySession(), viewport=(1440, 900))

    assert executor._scale_coordinates(500, 500) == (720.0, 450.0)
    assert executor._scale_coordinates(-1, 1001) == (0.0, 900)
