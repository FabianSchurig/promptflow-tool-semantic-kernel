import pytest
from promptflow_tool_semantic_kernel.tools.lights_plugin import LightsPlugin


@pytest.fixture
def lights_plugin():
    return LightsPlugin()


def test_get_state(lights_plugin):
    expected = [
        {
            "id": 1,
            "name": "Table Lamp",
            "is_on": False
        },
        {
            "id": 2,
            "name": "Porch light",
            "is_on": False
        },
        {
            "id": 3,
            "name": "Chandelier",
            "is_on": True
        },
    ]
    assert lights_plugin.get_state() == expected


def test_change_state_turn_on(lights_plugin):
    light_id = 1
    new_state = True
    expected = {"id": 1, "name": "Table Lamp", "is_on": True}
    assert lights_plugin.change_state(light_id, new_state) == expected
    assert lights_plugin.get_state()[0]["is_on"] == True


def test_change_state_turn_off(lights_plugin):
    light_id = 3
    new_state = False
    expected = {"id": 3, "name": "Chandelier", "is_on": False}
    assert lights_plugin.change_state(light_id, new_state) == expected
    assert lights_plugin.get_state()[2]["is_on"] == False


def test_change_state_invalid_id(lights_plugin):
    light_id = 99
    new_state = True
    assert lights_plugin.change_state(light_id, new_state) == None
