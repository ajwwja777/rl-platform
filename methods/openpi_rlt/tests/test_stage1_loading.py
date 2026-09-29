"""Inference restores only used parameters, with unchanged values and outputs."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import inference_tree


def test_unknown_checkpoint_branches_fail_closed():
    with pytest.raises(ValueError, match="roots"):
        inference_tree({"other_model": {}})
    with pytest.raises(ValueError, match="encoder"):
        inference_tree({"vla": {}, "rlt_module": {"decoder": {}}})
    with pytest.raises(ValueError, match="encoder"):
        inference_tree({"vla": {}, "rlt_module": {"encoder": {}, "unexpected": {}}})


def test_restore_skips_decoder_io_and_preserves_source(tmp_path, monkeypatch):
    # This test runs in the frozen Stage-1 environment, never on a robot.
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from orbax.checkpoint._src.serialization import type_handlers
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import restore_inference_params

    params = {"vla": {"kernel": {"value": np.arange(15, dtype=np.float32).reshape(3, 5)}},
              "rlt_module": {
                  "encoder": {"kernel": {"value": np.linspace(-1, 1, 11, dtype=np.float32)}},
                  "decoder": {"unused": {"value": np.ones(100, dtype=np.float32)}}}}
    path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})
    def hashes():
        return {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in path.rglob("*") if f.is_file()}
    before = hashes()
    restored_names = []
    original = type_handlers.NumpyHandler.deserialize
    async def tracked(self, infos, args=None):
        restored_names.extend(info.name for info in infos)
        return await original(self, infos, args)
    monkeypatch.setattr(type_handlers.NumpyHandler, "deserialize", tracked)
    loaded = restore_inference_params(path, restore_type=np.ndarray)
    assert restored_names and all("decoder" not in name for name in restored_names)
    assert set(loaded["rlt_module"]) == {"encoder"}
    np.testing.assert_array_equal(loaded["vla"]["kernel"],
                                  params["vla"]["kernel"]["value"].astype(jnp.bfloat16))
    np.testing.assert_array_equal(loaded["rlt_module"]["encoder"]["kernel"],
                                  params["rlt_module"]["encoder"]["kernel"]["value"].astype(jnp.bfloat16))
    assert before == hashes()


def test_abstract_encoder_matches_original_full_model():
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax.nnx.bridge import ToNNX
    from openpi.models.rl_token import RLTokenConfig, RLTokenModel

    config = RLTokenConfig(num_layers=1, input_dim=8, embed_dim=8, num_heads=2)
    prefix = jnp.arange(40, dtype=jnp.float32).reshape(1, 5, 8) / 40
    mask = jnp.ones((1, 5), dtype=jnp.bool_)
    full = ToNNX(RLTokenModel(config=config))
    full.lazy_init(prefix, mask, rngs=nnx.Rngs(42), train=False)
    _, original = nnx.split(full)

    def build():
        model = ToNNX(RLTokenModel(config=config))
        model.lazy_init(prefix, mask, rngs=nnx.Rngs(42), method="encode", train=False)
        return model

    abstract = nnx.eval_shape(build)
    graph, state = nnx.split(abstract)
    assert set(state.to_pure_dict()) == {"encoder"}
    assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(state.to_pure_dict()))
    state.replace_by_pure_dict({"encoder": original.to_pure_dict()["encoder"]})
    restored = nnx.merge(graph, state)
    np.testing.assert_array_equal(
        np.asarray(full(prefix, mask, method="encode", train=False)),
        np.asarray(restored(prefix, mask, method="encode", train=False)),
    )
