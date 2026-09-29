"""Resolve trusted project registrations without importing model frameworks."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def resolve(model_id,root=ROOT):
    root=Path(root)
    models=json.loads((root/"configs/deployment_models.json").read_text())["models"]
    row=next((r for r in models if r["id"]==model_id),{})
    name=row.get("runtime_profile")
    if not name:return None
    registry=json.loads((root/"configs/experiments/runtime_profiles.json").read_text())
    profile=registry["profiles"][name]
    result={"name":name,"experiment":profile["experiment"]}
    for key in ("config","run_dir"):
        path=(root/profile[key]).resolve()
        if root.resolve() not in path.parents:raise ValueError("Runtime profile escapes project root")
        result[key]=str(path)
    if not Path(result["config"]).is_file():raise ValueError("Prepare candidate before loading")
    return result
if __name__=="__main__":
    import sys
    result=resolve(sys.argv[1] if len(sys.argv)>1 else "")
    if result:
        print(result["experiment"]);print(result["config"]);print(result["run_dir"])
