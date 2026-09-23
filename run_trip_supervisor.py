"""
Run the TripSupervisor scenario end-to-end.

Examples:
    # Clean run, no injected failures, races possible on the shared budget
    python run_trip_supervisor.py

    # Safe blackboard as a control (no races possible)
    python run_trip_supervisor.py --blackboard-mode safe

    # Induce specific MAST modes + a hallucination point
    python run_trip_supervisor.py --enable fm_2_4_information_withholding,fm_3_2_incomplete_verification,hallucinate_prompt_pressure

Requires Ollama running locally with the model in trip_workflow.MODEL pulled
(default: qwen3:8b).
"""
import argparse
import asyncio
import json
import uuid

from trip_failure_config import FailureConfig
from trip_workflow import build_trip_workflow
from trip_hallucination import check_hallucination


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable", type=str, default="",
        help="Comma-separated FailureConfig field names to set True, e.g. "
             "fm_2_4_information_withholding,fm_3_2_incomplete_verification",
    )
    parser.add_argument("--blackboard-mode", choices=["safe", "unsafe"], default="unsafe")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def build_config(args) -> FailureConfig:
    cfg = FailureConfig(blackboard_mode=args.blackboard_mode)
    for flag in (f.strip() for f in args.enable.split(",") if f.strip()):
        if not hasattr(cfg, flag):
            raise ValueError(f"Unknown failure flag: {flag}")
        setattr(cfg, flag, True)
    return cfg


async def main():
    args = parse_args()
    cfg = build_config(args)

    nodes, blackboard = build_trip_workflow(cfg)
    task_id = str(uuid.uuid4())

    from workflow import WorkflowEngine
    engine = WorkflowEngine(nodes, blackboard, max_concurrency=4,
                             log_path=f"{args.log_dir}/trip_workflow_{task_id[:8]}.jsonl")

    print(f"=== TripSupervisor run {task_id[:8]} ===")
    print(f"Induced failure flags: {cfg.enabled_flags() or '(none)'}")
    print(f"Blackboard mode: {cfg.blackboard_mode}\n")

    results = await engine.run(task_id)

    print("--- Per-agent results ---")
    for node_id, r in results.items():
        print(f"[{node_id}] status={r.status} latency={r.latency_s:.2f}s attempt={r.attempt}")
        if r.status != "success":
            print(f"    FAILED: {r.error}")

    final_record = blackboard.snapshot().get("final_record", {})
    digest_text = blackboard.snapshot().get("digest_text", "")

    print("\n--- Final record (ground truth) ---")
    print(json.dumps(final_record, indent=2, default=str))

    print("\n--- Trip Digest output ---")
    print(digest_text)

    print("\n--- Hallucination check ---")
    hall = check_hallucination(final_record, digest_text)
    print(json.dumps(hall, indent=2))

    print("\n--- Blackboard race detection (budget_remaining accumulator) ---")
    race = final_record.get("budget_race_check", {})
    print(json.dumps(race, indent=2))
    if race.get("race_detected"):
        print("  -> LOST UPDATE: at least one agent's budget subtraction used a stale base value.")

    blackboard.dump_history(f"{args.log_dir}/trip_blackboard_{task_id[:8]}.jsonl")

    print(f"\nTask success: {final_record.get('task_success')}")
    print(f"Logs: {args.log_dir}/trip_workflow_{task_id[:8]}.jsonl , "
          f"{args.log_dir}/trip_blackboard_{task_id[:8]}.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
