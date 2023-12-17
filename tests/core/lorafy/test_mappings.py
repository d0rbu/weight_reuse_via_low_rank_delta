from core.lorafy.mappings import layer_mappings


def test_single_base_mapping():
    expected_single_base_mapping = [
        {1: 0, 2: 0, 3: 0},
        {0: 1, 2: 1, 3: 1},
        {0: 2, 1: 2, 3: 2},
        {0: 3, 1: 3, 2: 3},
    ]

    actual_single_base_mapping = list(layer_mappings(4, 1))

    assert actual_single_base_mapping == expected_single_base_mapping

def test_double_base_mapping():
    expected_double_base_mapping = [
        {2: 0},
        {2: 1},
        {1: 0},
        {1: 2},
        {0: 1},
        {0: 2},
    ]

    actual_double_base_mapping = list(layer_mappings(3, 2))
    print(actual_double_base_mapping)

    assert actual_double_base_mapping == expected_double_base_mapping
