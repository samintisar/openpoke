import pytest

from evals.systems import SYSTEMS, get_system


def test_options_reach_the_system():
    assert get_system("retrieved", {"search": False}).search is False
    assert get_system("retrieved").search is True


def test_unknown_options_are_refused():
    with pytest.raises(ValueError, match="does not take options"):
        get_system("baseline", {"summaries": False})


def test_a_type_error_inside_a_constructor_is_not_masked(monkeypatch):
    class Broken:
        def __init__(self, search=True):
            raise TypeError("real bug")

    monkeypatch.setitem(SYSTEMS, "broken", Broken)
    with pytest.raises(TypeError, match="real bug"):
        get_system("broken", {"search": False})
