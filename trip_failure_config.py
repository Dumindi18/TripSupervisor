"""
Toggleable failure-injection flags for the TripSupervisor scenario.

Each flag is a specific, minimal mechanism for inducing one MAST failure
mode (or one of the FC4 concurrency modes / the hallucination point). Flags
are independent and combinable. Whatever set is True for a run is your
"silver" ground-truth label for what was intentionally induced -- compare
this against what your LLM annotator (built later) actually detects.
"""
from dataclasses import dataclass, asdict


@dataclass
class FailureConfig:
    # ---- FC1: System Design Issues -----------------------------------
    fm_1_1_disobey_task_spec: bool = False
    # Flight Agent told there's no hard budget limit -> may pick the priciest option.

    fm_1_2_disobey_role_spec: bool = False
    # Activity Agent is told it may recommend changing the flight -- outside its role.

    fm_1_4_loss_of_history: bool = False
    # Checkout prompt omits the original customer brief (preferences/destination).

    fm_1_5_unaware_termination: bool = False
    # Verifier is not told it may only check once; combined with no max-iteration
    # guard in the calling code this models a system with no stop condition.
    # (The demo harness still caps retries for safety -- see run_trip_supervisor.py.)

    # ---- FC2: Inter-Agent Misalignment --------------------------------
    fm_2_2_fail_to_clarify: bool = False
    # Hotel Agent told to proceed on its own interpretation of "free breakfast"
    # rather than asking the supervisor to clarify.

    fm_2_3_task_derailment: bool = False
    # Activity Agent told it may also suggest shopping/souvenirs, beyond its task.

    fm_2_4_information_withholding: bool = False
    # Hotel Agent told not to mention the loyalty_id requirement to the supervisor.

    fm_2_5_ignored_other_agent_input: bool = False
    # Verifier prompt is only given the flight + hotel bookings, not activity --
    # it will "ignore" the activity agent's contribution when checking budget.

    fm_2_6_reasoning_action_mismatch: bool = False
    # Flight Agent told to reason about the cheapest option but then book the
    # fastest-departing one regardless of what its own reasoning concluded.

    # ---- FC3: Task Verification ----------------------------------------
    fm_3_1_premature_termination: bool = False
    # Checkout finalizes the booking even when the verifier reported issues.

    fm_3_2_incomplete_verification: bool = False
    # Verifier only checks that confirmation numbers exist, not budget compliance.

    fm_3_3_incorrect_verification: bool = False
    # Verifier checks against the *original* budget_cap instead of the live,
    # already-decremented budget_remaining -- a stale-reference check.

    # ---- FC4: Parallel-execution / concurrency failures -----------------
    blackboard_mode: str = "unsafe"  # "unsafe" (races possible) or "safe" (control)

    # ---- Hallucination induction (Trip Digest) --------------------------
    hallucinate_drop_context: bool = False
    # Trip Digest is fed a final_record with fields redacted, encouraging
    # it to fill gaps with invented specifics.

    hallucinate_prompt_pressure: bool = False
    # Trip Digest is told to "make it sound exciting" and to always mention a
    # loyalty/rewards note, even when none exists in the record.

    def as_dict(self):
        return asdict(self)

    def enabled_flags(self):
        return [k for k, v in self.as_dict().items() if v is True]
