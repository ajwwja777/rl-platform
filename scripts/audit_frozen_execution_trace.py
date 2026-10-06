#!/usr/bin/env python3
"""Read-only numeric evidence to a new output; no robot/model/Replay dependency."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.openpi_rlt.experiments.frozen_trace_audit import audit_traces


def plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for index, ep in enumerate(report["episodes"]):
        records = ep["series"]["logical"]
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        for d, ax in enumerate(axes):
            for key, label in [("reference", "Reference (logical queue)"), ("proposal", "Actor queue target"),
                                ("action", "Logical actual action"), ("feedback", "Endpoint feedback")]:
                x = [r["row"] for r in records]
                y = [np.nan if r[key] is None else r[key][d]*1000 for r in records]
                if np.isfinite(y).any():
                    ax.plot(x, y, label=label, linewidth=1)
            for r in records:
                if r["human"]:
                    ax.axvspan(r["row"]-.5, r["row"]+.5, color="orange", alpha=.08)
            ax.set_ylabel(report["dimensions"][d] + " (" + report["dimension_units"][d] + ")")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="best", ncol=2)
        axes[-1].set_xlabel("Logical row index (no inferred semantic phases)")
        fig.suptitle(str(ep["episode_identity"]) + " | " + ep["split"] + " | " + str(ep["condition"]) + " | " + ep["outcome_class"])
        fig.tight_layout(rect=[0, 0, 1, .97])
        fig.savefig(output / ("episode-%03d-actions.png" % index), dpi=140)
        plt.close(fig)
        physical = ep["series"]["physical"]
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        for d, ax in enumerate(axes):
            if physical:
                origin = physical[0]["t"]
                ax.plot([r["t"]-origin for r in physical],
                        [r["action"][d]*1000 for r in physical], ".-", ms=2,
                        label="Physical published command")
                points = [r for r in records if r["feedback"] is not None and r["t"] is not None]
                ax.plot([r["t"]-origin for r in points],
                        [r["feedback"][d]*1000 for r in points], ".", label="Logical endpoint feedback")
            else:
                ax.text(.5,.5,"Missing physical receipts",ha="center",transform=ax.transAxes)
            ax.set_ylabel(report["dimensions"][d]+" ("+report["dimension_units"][d]+")")
            ax.grid(alpha=.2)
        if physical:
            axes[0].legend(loc="best")
        axes[-1].set_xlabel("Seconds since first physical receipt; pauses/gaps retained")
        fig.suptitle("Physical command / endpoint feedback | "+str(ep["episode_identity"])+" | "+ep["split"])
        fig.tight_layout(rect=[0,0,1,.97])
        fig.savefig(output / ("episode-%03d-physical.png" % index), dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        if len(physical) > 1:
            t = np.array([r["t"] for r in physical])
            axes[0].plot(t[1:]-t[0], np.diff(t)*1000, ".-", label="All receipt intervals, including pauses/gaps")
            axes[0].legend()
        else:
            axes[0].text(.5,.5,"Physical publication receipts missing",ha="center",transform=axes[0].transAxes)
        for ev in ep["series"]["inference_events"]:
            start, end = ev.get("request_started_monotonic"), ev.get("request_finished_monotonic")
            if start is not None and end is not None:
                axes[1].plot([start, end], [ev.get("logical_step"), ev.get("logical_step")], "o-")
        axes[0].set(xlabel="Seconds since first receipt", ylabel="Interval (ms)")
        axes[1].set(xlabel="Request monotonic time (s)", ylabel="Inference anchor logical step")
        fig.suptitle("Diagnostic timing | " + str(ep["episode_identity"]) + " | " + ep["split"])
        fig.tight_layout(rect=[0,0,1,.95])
        fig.savefig(output / ("episode-%03d-timing.png" % index), dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; never overwrite existing evidence")
    paths = [p.resolve() for p in args.trace]
    output = args.output.resolve()
    if any(output == p or output in p.parents for p in paths):
        parser.error("Output must not contain or replace source traces")
    identities = {}
    if args.manifest:
        value = json.loads(args.manifest.read_text())
        for item in value["episodes"]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = args.manifest.parent / path
            key = str(path.resolve())
            if key in identities:
                parser.error("Duplicate trace manifest identity")
            identities[key] = item
    report = audit_traces(paths, identities)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.plots:
        plots(report, output)
    print(json.dumps(dict(status=report["status"], episodes=len(report["episodes"]), output=str(output))))

if __name__ == "__main__":
    main()
