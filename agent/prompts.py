"""加载主 Agent 与子 Agent 的 YAML 提示词配置。"""

import yaml
from pathlib import Path

def load_yaml(file_path):
    """安全加载 UTF-8 YAML 文件。"""
    with open(file_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

project_root_path = Path(__file__).parents[1]
yaml_file_path = project_root_path / "prompt" / "prompts.yml"

prompt_yaml_content = load_yaml(yaml_file_path)

main_agent_content = prompt_yaml_content["main_agent"]

sub_agents_content = prompt_yaml_content["sub_agents"]
