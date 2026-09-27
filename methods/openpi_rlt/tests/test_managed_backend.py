
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def test_managed_interfaces_require_unified_console_and_remain_disarmed():
    for name in ("interface_task2_teach_rlt_online.sh", "interface_task2_teach_rlt_continue5416.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "--managed-backend" in text
        assert "--lifecycle-state" in text
        assert "8015/api/rlt-recorder" in text
        assert "ready_disarmed" in text
        assert 'if [[ "$MANAGED_BACKEND" == "0" ]]' in text
        assert "lifecycle_cli.py" in text
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)


def test_no_record_interfaces_require_frozen_actor_before_starting_services():
    for name in ("interface_task2_teach_rlt_online.sh", "interface_task2_teach_rlt_continue5416.sh"):
        result = subprocess.run(["bash", str(ROOT / "scripts" / name), "4000", "--no-record"], capture_output=True, text=True)
        assert result.returncode == 2
        assert "requires --frozen-actor" in result.stderr
