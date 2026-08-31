from pathlib import Path

import yaml

from schema import DeviceProfile, ModelTopology, OptimizationStep, Recipe, _hash_dict


class RecipeEngine:
    def generate(
        self,
        topology: ModelTopology,
        device: DeviceProfile,
        steps: list[OptimizationStep],
    ) -> Recipe:
        return Recipe(
            version="1.0",
            model_id=topology.model_id,
            topology_hash=_hash_dict(topology.model_dump()),
            device_hash=_hash_dict(device.model_dump()),
            steps=steps,
        )

    def save(self, recipe: Recipe, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(recipe.model_dump(mode="json"), f, default_flow_style=False)

    def load(self, path: Path) -> Recipe:
        with open(path) as f:
            data = yaml.safe_load(f)
        return Recipe(**data)
