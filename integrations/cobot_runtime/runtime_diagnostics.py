"""Read-only RLT failure classification shared by operator clients; stdlib only."""
import re

def classify_failure(output):
    # The final supervisor exception is a consequence. Preserve the earlier cause.
    causes = [
        ("rtc_delay_exceeded", ("actual delay exceeded predicted delay", "RTC delay exceeded budget")),
        ("execution_clock_late", ("Execution clock missed its deadline",)),
        ("observation_stale", ("Control observations stale", "stale control", "Stale control")),
        ("gpu_out_of_memory", ("RESOURCE_EXHAUSTED", "CUDA out of memory",)),
        ("execution_failed", ("execution_timing_failed:",)),
    ]
    for code, markers in causes:
        for line in reversed(output.splitlines()):
            if any(marker in line for marker in markers):
                return {"code": code, "cause": line.strip()[:600]}
    lines = [line.strip() for line in output.splitlines()
             if re.match(r"^(?:[\w.]+Error|[\w.]+Exception):", line.strip())]
    return {"code": "runtime_process_exited",
            "cause": next((x for x in reversed(lines) if "exited with code" not in x),
                          lines[-1] if lines else "RLT runtime process exited; inspect its log")}
