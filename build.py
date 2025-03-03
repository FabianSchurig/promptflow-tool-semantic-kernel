# This file is executed during poetry build to generate tool metadata

import yaml
from pathlib import Path

PACKAGE_NAME = "promptflow_tool_semantic_kernel"


def generate_tool_meta_cache():
    """Generate tool meta cache file in package."""
    tools_meta_path = Path(PACKAGE_NAME) / "yamls"
    tool_meta_infos = []

    if tools_meta_path.exists():
        for yaml_file in tools_meta_path.glob("*.yaml"):
            with open(yaml_file, "r") as f:
                tool_meta = yaml.safe_load(f)
                tool_meta_infos.append(tool_meta)

    # Create meta cache file
    cache_file_path = Path(PACKAGE_NAME) / ".tool_meta_cache.yaml"
    with open(cache_file_path, "w") as f:
        yaml.dump(tool_meta_infos, f)

    print(f"Tool meta cache file generated at {cache_file_path}")


# Execute when this script is run
if __name__ == "__main__":
    generate_tool_meta_cache()
