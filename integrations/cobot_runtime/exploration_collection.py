"""Explicit frozen-Actor collection contract; no model or service is loaded here."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

MODEL_ID = "plug-v3-hil8-explore"
SMOOTH_MODEL_ID = "plug-v3-hil8-explore-smooth"
ACTOR_SHA256 = "020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5"
STATS_SHA256 = "4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c"


def selected(environ=None):
    env = os.environ if environ is None else environ
    return env.get("COBOT_DEPLOYMENT_MODEL_ID") in (MODEL_ID, SMOOTH_MODEL_ID)


def locations():
    from .paths import RLT, DATA
    suffix = "hil8-7281-smooth-v1" if os.environ.get("COBOT_DEPLOYMENT_MODEL_ID") == SMOOTH_MODEL_ID else "hil8-7281-v1"
    return dict(run=RLT / ("outputs/exploration-collection/"+suffix),
                recording=DATA / ("datasets/plug_insertion/recordings/rl-platform/rlt/exploration-"+suffix))



def recording_root(phase):
    """Honor the web selection at each Episode; private path is only a fallback.

    Recording location and Replay journal isolation are independent contracts.
    Never override an operator-selected directory to enforce Replay isolation.
    """
    from .profile_storage import settings, selected_root, validate_root
    choices = settings()
    if choices.get("selected") or (choices.get("current") or {}).get(phase):
        return selected_root(phase)
    return validate_root(locations()["recording"])


def verify_assets(snapshot, stats):
    for name, path, expected in (("Actor", snapshot, ACTOR_SHA256), ("normalization", stats, STATS_SHA256)):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Exploration collection " + name + " SHA256 mismatch")


def validate_launch(mode, environ=None):
    env = os.environ if environ is None else environ
    if not selected(env):
        return
    if mode != "frozen" or env.get("COBOT_RLT_SHARED_MODEL") != "1":
        raise ValueError("Exploration collection requires frozen shared-model mode")
    if any(env.get(key) for key in ("COBOT_RLT_BRANCH_CONFIG", "COBOT_RLT_ONLINE_SEED",
                                   "COBOT_RLT_EXPERIMENT_PROFILE", "COBOT_RLT_SUPPORTED_PROFILE")):
        raise ValueError("Exploration collection cannot inherit a training branch")
    from methods.openpi_rlt.cobot_adapter.execution_profiles import selected_profile
    profile = selected_profile(environ=env)
    smooth = env.get("COBOT_DEPLOYMENT_MODEL_ID") == SMOOTH_MODEL_ID
    if profile is None or profile[1].publish_hz != 50 or not profile[1].rtc or profile[1].smoothing_tau_sec != (.12 if smooth else .08) or profile[1].smoothing_order != (2 if smooth else 1):
        raise ValueError("Exploration collection requires 50 Hz, RTC and the selected entry filter")
    snapshot = Path(env["COBOT_EVAL_SNAPSHOT"])
    verify_assets(snapshot, snapshot.parent.parent / "action_norm_stats.json")


def configure(config, mode):
    if not selected():
        return config
    if mode != "frozen":
        raise ValueError("Exploration collection cannot train")
    runtime = config["runtime"]
    verify_assets(runtime["actor_service"]["snapshot_path"], config["experiment"]["rl"]["action_norm_stats_path"])
    run = locations()["run"]
    runtime["replay"]["journal_path"] = str(run / "replay/collection.pkl")
    runtime["learner_service"]["checkpoint_dir"] = str(run / "unused-checkpoints")
    runtime["learner_service"]["actor_snapshot_path"] = str(run / "unused-actor/actor_snapshot.pkl")
    runtime["env_driver"]["actor_deterministic"] = True
    runtime["env_driver"]["safe_fallback_to_ref"] = False
    runtime["monitoring"]["enable_wandb"] = False
    return config


def validate_runtime(config_path, environ=None):
    env = os.environ if environ is None else environ
    if not selected(env):
        return
    import yaml
    if env.get("RLT_DISABLE_LEARNER") != "1" or env.get("COBOT_RLT_DISABLE_PHASE_CONTROLLER") != "1":
        raise ValueError("Exploration collection must disable Learner and phase controller")
    for key, value in {"COBOT_RLT_HOLD_RIGHT_GRIPPER": "1", "COBOT_RLT_REPLAY_ACTION_PRECISION": "float32",
                       "COBOT_RLT_HIL_SAMPLING": "logical20", "COBOT_RLT_INPUT_AUDIT": "strict",
                       "COBOT_RLT_RAW_OBSERVATION_CONTRACT": "trace"}.items():
        if env.get(key) != value:
            raise ValueError("Exploration collection missing required " + key)
    config = yaml.safe_load(Path(config_path).read_text())
    runtime = config["runtime"]
    run = locations()["run"]
    verify_assets(runtime["actor_service"]["snapshot_path"], config["experiment"]["rl"]["action_norm_stats_path"])
    if runtime["replay"]["journal_path"] != str(run / "replay/collection.pkl"):
        raise ValueError("Exploration collection must use its private Replay")
    if not runtime["env_driver"]["actor_deterministic"] or runtime["env_driver"]["safe_fallback_to_ref"]:
        raise ValueError("Exploration requires deterministic frozen Actor without fallback")
    if runtime["learner_service"]["actor_snapshot_path"] != str(run / "unused-actor/actor_snapshot.pkl"):
        raise ValueError("Exploration must never write the served Actor")
    if env.get("RLT_OUTPUT_DIR") != str(run / "online") or env.get("COBOT_RLT_TRACE_DIR") != str(run / "traces"):
        raise ValueError("Exploration evidence must use private output directories")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("launch", "validate"))
    parser.add_argument("value")
    args = parser.parse_args()
    if args.operation == "validate":
        validate_runtime(args.value)
    elif selected():
        validate_launch(args.value)
        print(locations()["run"])
        # Managed collection entries advertise their own directory at launch.
        # This matches the installed model-storage behavior, not an old test selection.
        print(locations()["recording"])
