from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Recipe:
    name: str
    title: str
    description: str
    author: str
    prompts: List[str]


_BUILTIN: List[Recipe] = [
    Recipe(
        name="search-learn",
        title="Search & Learn (loop-enforced)",
        description="Workflow prompt pack for evidence-backed Q&A with persisted ledgers, hypothesis testing, and hallucination detection.",
        author="Berry",
        prompts=["search_and_learn_verified"],
    ),
    Recipe(
        name="boilerplate-verified",
        title="Generate boilerplate/content (verified)",
        description="Workflow prompt pack for generating tests/docs/config with an auditable trace budget check.",
        author="Berry",
        prompts=["generate_boilerplate_verified"],
    ),
    Recipe(
        name="inline-completion-guard",
        title="Inline completion guard (verified)",
        description="Workflow prompt pack for reviewing tab-complete suggestions using an audited micro-trace.",
        author="Berry",
        prompts=["inline_completion_review"],
    ),
    Recipe(
        name="greenfield-prototype",
        title="Greenfield prototyping (loop-enforced)",
        description="Workflow prompt pack for fast prototyping with persisted ledgers, explicit facts/decisions/assumptions, and evidence loops.",
        author="Berry",
        prompts=["greenfield_prototyping_verified"],
    ),
    Recipe(
        name="objective-optimization",
        title="Objective Optimization Agent",
        description="Workflow prompt pack for optimizing any well-defined objective with baseline measurement, keep/revert decisions, and verified claims.",
        author="Berry",
        prompts=["objective_optimization_agent"],
    ),
]


def builtin_recipes() -> List[Recipe]:
    return list(_BUILTIN)


def get_builtin_recipe(name: str) -> Optional[Recipe]:
    for r in _BUILTIN:
        if r.name == name:
            return r
    return None


def project_recipes_dir(project_root: Path) -> Path:
    return Path(project_root) / ".berry" / "recipes"


def list_project_recipes(project_root: Path) -> List[Path]:
    d = project_recipes_dir(project_root)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.json") if p.is_file())


def install_recipe_to_project(recipe: Recipe, *, project_root: Path, force: bool = False) -> Path:
    d = project_recipes_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{recipe.name}.json"
    if out.exists() and not force:
        raise FileExistsError(f"Recipe already exists: {out}")
    out.write_text(json.dumps(asdict(recipe), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _validate_recipe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Recipe payload must be an object")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Recipe missing name")
    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("Recipe missing title")
    description = str(payload.get("description") or "").strip()
    if not description:
        raise ValueError("Recipe missing description")
    author = str(payload.get("author") or "").strip()
    if not author:
        raise ValueError("Recipe missing author")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("Recipe prompts must be a list of non-empty strings")
    return {
        "name": name,
        "title": title,
        "description": description,
        "author": author,
        "prompts": prompts,
    }


def install_recipe_file_to_project(path: Path, *, project_root: Path, force: bool = False) -> Path:
    payload = load_recipe_file(path)
    payload = _validate_recipe_payload(payload)
    d = project_recipes_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{payload['name']}.json"
    if out.exists() and not force:
        raise FileExistsError(f"Recipe already exists: {out}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_recipe_file(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_recipes(recipes: Iterable[Recipe], out_path: Path) -> Path:
    out_path.write_text(
        json.dumps([asdict(r) for r in recipes], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
