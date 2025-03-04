from typing import Annotated
from semantic_kernel.functions import kernel_function


class LightsPlugin:
    lights = [
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

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(
        self, id: Annotated[int, "The ID of the light to get its state"],
        all: Annotated[bool, "Flag to get all lights"]
    ) -> Annotated[list[dict], "A list of lights with their properties"]:
        """
        Gets a list of lights and their current state.

        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a light
                 with properties such as id, name, and is_on state.
        """
        if id:
            for light in self.lights:
                if light["id"] == id:
                    return [light]
            return []
        if all:
            return self.lights
        return []

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(
        self,
        id: Annotated[int, "The ID of the light to change"],
        is_on: Annotated[bool, "Whether to turn the light on or off"],
    ) -> str:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return light
        return "Light state changed successfully"
