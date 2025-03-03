import importlib
import importlib.metadata

# Define the entry point group name
PACKAGE_TOOLS_ENTRY = "package_tools"


def test():
    """List all package tools information using the `package-tools` entry point.

    This function iterates through all entry points registered under the group "package_tools."
    For each tool, it imports the associated module to ensure its validity and then prints
    information about the tool.

    Note:
    - Make sure your package is correctly packed to appear in the list.
    - The module is imported to validate its presence and correctness.

    Example of tool information printed:
    ----identifier
    {'module': 'module_name', 'package': 'package_name', 'package_version': 'package_version', ...}
    """
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        print("entry_points.select(group=PACKAGE_TOOLS_ENTRY)")
        entry_points = entry_points.select(group=PACKAGE_TOOLS_ENTRY)
        print(entry_points)
    else:
        entry_points = entry_points.get(PACKAGE_TOOLS_ENTRY, [])
    for entry_point in entry_points:
        list_tool_func = entry_point.load()
        package_tools = list_tool_func()

        for identifier, tool in package_tools.items():
            importlib.import_module(
                tool["module"])  # Import the module to ensure its validity
            print(f"----{identifier}\n{tool}")


if __name__ == "__main__":
    test()
