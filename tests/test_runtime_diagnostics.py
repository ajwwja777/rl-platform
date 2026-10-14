from integrations.cobot_runtime.runtime_diagnostics import classify_failure

def test_root_cause_not_supervisor_summary():
    value=classify_failure("RTCActionStateError: actual delay exceeded predicted delay\n"
                           "RuntimeError: env_driver exited with code 1")
    assert value["code"]=="rtc_delay_exceeded"
    assert "actual delay" in value["cause"]

def test_unknown_failure_retains_error():
    value=classify_failure("ValueError: malformed observation\n"
                           "RuntimeError: env_driver exited with code 1")
    assert value["code"]=="runtime_process_exited"
    assert value["cause"]=="ValueError: malformed observation"

def test_oom_never_mislabeled_recorder_failure():
    assert classify_failure("RuntimeError: CUDA out of memory")["code"]=="gpu_out_of_memory"


def test_learner_segfault_is_not_ros_teardown():
    value = classify_failure(
        "RuntimeError: ROS shut down before Cobot RLT was armed\n"
        "RuntimeError: learner_service exited with code -11")
    assert value == {"code": "runtime_worker_crashed",
                     "cause": "learner_service exited with code -11"}


def test_orderly_sigterm_is_not_native_crash():
    value = classify_failure(
        "ValueError: bad configuration\n"
        "RuntimeError: learner_service exited with code -15")
    assert value["code"] == "runtime_process_exited"
    assert value["cause"] == "ValueError: bad configuration"


def test_specific_oom_remains_primary_to_abort():
    value = classify_failure(
        "RuntimeError: CUDA out of memory\n"
        "RuntimeError: learner_service exited with code -6")
    assert value["code"] == "gpu_out_of_memory"
