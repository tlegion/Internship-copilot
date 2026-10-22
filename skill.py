from pathlib import Path
import re

import yaml

from errors import ToolError

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def discover_skills(skills_dir: Path = SKILLS_DIR) -> dict:
    root = Path(skills_dir).resolve()
    catalog = {}
    for path in sorted(root.glob("*/SKILL.md")):
        if not path.resolve().is_relative_to(root):
            raise ToolError(f"Skill path is outside the Skills directory: {path}")
        try:
            with path.open(encoding="utf-8") as source:
                if source.readline().rstrip("\r\n") != "---":
                    raise ToolError(f"Missing YAML frontmatter: {path}")
                lines = []
                for line in source:
                    if line.rstrip("\r\n") == "---":
                        break
                    lines.append(line)
                else:
                    raise ToolError(f"Unclosed YAML frontmatter: {path}")
            metadata = yaml.safe_load("".join(lines))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ToolError(f"Unable to discover Skill {path}: {error}") from error

        if not isinstance(metadata, dict):
            raise ToolError(f"Skill frontmatter must be a mapping: {path}")
        name = metadata.get("name")
        description = metadata.get("description")
        if (not isinstance(name, str) or len(name) > 64
                or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
                or name != path.parent.name):
            raise ToolError(f"Invalid Skill name or directory mismatch: {path}")
        if not isinstance(description, str) or not description.strip() or len(description) > 1024:
            raise ToolError(f"Invalid Skill description: {path}")
        catalog[name] = {"description": description, "path": path, "frontmatter": metadata}
    return catalog


def load_skill(name: str, catalog: dict) -> str:
    if not isinstance(name, str) or name not in catalog:
        raise ToolError(f"Unknown Skill: {name}")
    path = catalog[name]["path"]
    if not path.resolve().is_relative_to(path.parent.resolve()):
        raise ToolError(f"Skill instructions are outside the Skill directory: {name}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ToolError(f"Unable to load Skill {name}: {error}") from error


def read_skill_resource(name: str, path: str, catalog: dict, loaded_skills: set) -> str:
    if not isinstance(name, str) or name not in catalog:
        raise ToolError(f"Unknown Skill: {name}")
    if name not in loaded_skills:
        raise ToolError(f"Skill is not loaded: {name}")
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ToolError("Skill resource path must be a nonempty relative path")
    root = catalog[name]["path"].parent.resolve()
    resource = (root / path).resolve()
    if not resource.is_relative_to(root):
        raise ToolError("Skill resource path is outside the Skill directory")
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ToolError(f"Unable to read Skill resource {name}/{path}: {error}") from error
