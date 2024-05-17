import pytest
from musclebeachtools import get_group_value

def test_get_group_value():
    # Test with typical values
    assert get_group_value(5, 3) == 2
    assert get_group_value(10, 4) == 2
    assert get_group_value(8, 2) == 0

    # Test with channel equal to 0
    assert get_group_value(0, 5) == 0

    # Test with channel less than group_size
    assert get_group_value(2, 5) == 2

    # Test with channel equal to a multiple of group_size
    assert get_group_value(6, 3) == 0
    assert get_group_value(12, 4) == 0

    # Test with larger channel values
    assert get_group_value(100, 6) == 4
    assert get_group_value(101, 10) == 1

    # # Edge case: channel is negative
    # with pytest.raises(ValueError):
    #     get_group_value(-1, 3)

    # # Edge case: group_size is zero
    # with pytest.raises(ZeroDivisionError):
    #     get_group_value(10, 0)

if __name__ == "__main__":
    pytest.main()

