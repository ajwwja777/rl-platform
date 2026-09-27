"""Apply the upstream writer's behavioral contract to the RLT cached writer."""
from pathlib import Path
import capture_core
import importlib.util

source = Path(capture_core.__file__).parents[1] / "tests/test_hdf5_writer.py"
exec(compile(source.read_text(), str(source), "exec"))
spec = importlib.util.spec_from_file_location(
    "rlt_cached_writer", Path(__file__).parents[1] / "scripts/rlt_cached_writer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Hdf5EpisodeWriter = module.RltCachedWriter
