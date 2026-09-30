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
