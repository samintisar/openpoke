import pytest

from evals.systems import get_system


def test_options_reach_the_system():
    assert get_system("retrieved", {"search": False}).search is False
    assert get_system("retrieved").search is True


def test_unknown_options_are_refused():
    with pytest.raises(ValueError, match="does not take options"):
        get_system("baseline", {"summaries": False})
