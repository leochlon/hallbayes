from __future__ import annotations

import json

from berry.recipes import (
    builtin_recipes,
    get_builtin_recipe,
    install_recipe_file_to_project,
    install_recipe_to_project,
    list_project_recipes,
)


def test_builtin_recipes_nonempty():
    assert builtin_recipes()


def test_install_recipe_to_project(tmp_repo):
    r = get_builtin_recipe("search-learn")
    assert r is not None
    out = install_recipe_to_project(r, project_root=tmp_repo)
    assert out.exists()
    files = list_project_recipes(tmp_repo)
    assert out in files


def test_install_recipe_file_to_project(tmp_repo, tmp_path):
    payload = {
        "name": "custom-recipe",
        "title": "Custom Recipe",
        "description": "A shared recipe for testing import.",
        "author": "Test",
        "prompts": ["search_and_learn_verified"],
    }
    src = tmp_path / "recipe.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = install_recipe_file_to_project(src, project_root=tmp_repo)
    assert out.exists()
    files = list_project_recipes(tmp_repo)
    assert out in files


def test_objective_optimization_recipe_exists():
    r = get_builtin_recipe("objective-optimization")
    assert r is not None
    assert r.prompts == ["objective_optimization_agent"]
