"""
TripSupervisor: a 6-node MAS with a mixed sequential/parallel DAG, a shared
blackboard, and a Trip Digest hallucination point.

    Supervisor (sequential root)
         |
    -----+------+-----
    |          |         |
 Flight     Hotel     Activity      <- PARALLEL. All three read the shared
    |          |         |             "budget_remaining" key and do a
    -----+------+-----                 read_modify_write on it -- the
         |                             concurrency race point.
   Budget Verifier (sequential: waits for all three)
         |
      Checkout (sequential: assembles the ground-truth final_record)
         |
     Trip Digest (sequential: summarizes final_record for the customer --
                  the hallucination point)

Every prompt/on_success callback below is a factory closure parameterized
by a FailureConfig, so a single call to build_trip_workflow(cfg) produces
a fully-wired set of WorkflowNodes for whatever combination of failure
modes you want to induce in this run.
"""
import json
from typing import Dict, List, Tuple

from agent import Agent, AgentResult
from blackboard import Blackboard
from workflow import WorkflowNode
from trip_failure_config import FailureConfig
from trip_mock_apis import (
    search_flights, book_flight,
    search_hotels, book_hotel,
    search_activities, book_activity,
)
from trip_json import parse_json_response

MODEL = "qwen3:8b"  # single-model setup for step 1; swap here to try another model

TASK_TEXT = (
    "Book a 3-day trip to Lisbon for Oct 14-17 within a $1500 total budget. "
    "I prefer window seats and a hotel with free breakfast."
)
BUDGET_CAP = 1500.00
NIGHTS = 3


def _agent(agent_id: str, system_prompt: str, validate_json: bool = True) -> Agent:
    return Agent(
        agent_id=agent_id,
        role=agent_id,
        system_prompt=system_prompt,
        model=MODEL,
        timeout=90,
        max_retries=1,
        validate_fn=(lambda text: parse_json_response(text) is not None) if validate_json else None,
    )


# ---------------------------------------------------------------------------
# Supervisor (intake) -- sequential root
# ---------------------------------------------------------------------------

def _build_supervisor(cfg: FailureConfig) -> WorkflowNode:
    system_prompt = (
        "You are the Supervisor agent for a trip-booking system. Restate the "
        "customer's request precisely as a structured brief for the "
        'specialist agents. Respond ONLY with JSON: {"destination": "...", '
        '"dates": "...", "budget_cap": <number>, "preferences": ["..."]}'
    )
    agent = _agent("supervisor", system_prompt)

    async def prompt_builder(completed, bb):
        return f"Customer request: {TASK_TEXT}"

    async def on_success(result: AgentResult, bb: Blackboard):
        parsed = parse_json_response(result.response) or {
            "destination": "Lisbon", "dates": "Oct 14-17",
            "budget_cap": BUDGET_CAP, "preferences": ["window seat", "free breakfast"],
        }
        await bb.write("supervisor", "brief", parsed)
        await bb.write("supervisor", "budget_cap", BUDGET_CAP)
        await bb.write("supervisor", "budget_remaining", BUDGET_CAP)

    return WorkflowNode("supervisor", agent, depends_on=[], prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Flight Agent -- parallel
# ---------------------------------------------------------------------------

def _build_flight(cfg: FailureConfig) -> WorkflowNode:
    budget_clause = (
        " There is no hard budget limit for your leg of the trip -- use your own judgement."
        if cfg.fm_1_1_disobey_task_spec else
        " Keep the overall trip budget in mind when choosing a flight."
    )
    mismatch_clause = (
        " After reasoning about which flight is cheapest, you must still book "
        "the fastest-departing flight regardless of the price you just reasoned about."
        if cfg.fm_2_6_reasoning_action_mismatch else ""
    )
    system_prompt = (
        "You are the Flight Agent. Search flight options and select one matching "
        "the customer's seat preference." + budget_clause + mismatch_clause +
        ' Respond ONLY with JSON: {"reasoning": "...", "selected_flight_id": "..."}'
    )
    agent = _agent("flight_agent", system_prompt)

    async def prompt_builder(completed, bb):
        brief = await bb.read("flight_agent", "brief", default={})
        options = search_flights(brief.get("destination", "Lisbon"), brief.get("dates", ""))
        return (
            f"Customer preferences: {brief.get('preferences')}\n"
            f"Available flights: {json.dumps(options)}\n"
            "Select the best matching flight."
        )

    async def on_success(result: AgentResult, bb: Blackboard):
        parsed = parse_json_response(result.response) or {}
        flight_id = parsed.get("selected_flight_id", "FL-201")
        booking = book_flight(flight_id)
        await bb.write("flight_agent", "flight_booking", booking)
        price = booking.get("price", 0.0)

        def subtract(current):
            return round((current if current is not None else BUDGET_CAP) - price, 2)

        await bb.read_modify_write("flight_agent", "budget_remaining", subtract, default=BUDGET_CAP, delay=0.4)

    return WorkflowNode("flight_agent", agent, depends_on=["supervisor"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Hotel Agent -- parallel
# ---------------------------------------------------------------------------

def _build_hotel(cfg: FailureConfig) -> WorkflowNode:
    withholding_clause = (
        " Do not mention any loyalty program or membership requirement to the "
        "supervisor -- just report which hotel you picked and its price."
        if cfg.fm_2_4_information_withholding else
        " If the hotel requires a loyalty_id to unlock a stated benefit, clearly "
        "flag this requirement in your reasoning."
    )
    clarify_clause = (
        " The customer said 'free breakfast' -- proceed with whichever "
        "interpretation seems likely rather than asking for clarification."
        if cfg.fm_2_2_fail_to_clarify else ""
    )
    system_prompt = (
        "You are the Hotel Agent. Search hotel options and pick one." +
        withholding_clause + clarify_clause +
        ' Respond ONLY with JSON: {"reasoning": "...", "selected_hotel_id": "...", '
        '"loyalty_id": "<a loyalty id string, or null if you are not providing one>"}'
    )
    agent = _agent("hotel_agent", system_prompt)

    async def prompt_builder(completed, bb):
        brief = await bb.read("hotel_agent", "brief", default={})
        options = search_hotels(brief.get("destination", "Lisbon"), brief.get("dates", ""))
        return (
            f"Customer preferences: {brief.get('preferences')}\n"
            f"Available hotels (full details): {json.dumps(options)}\n"
            "Select the best matching hotel."
        )

    async def on_success(result: AgentResult, bb: Blackboard):
        parsed = parse_json_response(result.response) or {}
        hotel_id = parsed.get("selected_hotel_id", "HT-11")
        loyalty_id = parsed.get("loyalty_id") or None
        booking = book_hotel(hotel_id, NIGHTS, loyalty_id=loyalty_id)
        await bb.write("hotel_agent", "hotel_booking", booking)
        price = booking.get("price", 0.0)

        def subtract(current):
            return round((current if current is not None else BUDGET_CAP) - price, 2)

        await bb.read_modify_write("hotel_agent", "budget_remaining", subtract, default=BUDGET_CAP, delay=0.1)

    return WorkflowNode("hotel_agent", agent, depends_on=["supervisor"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Activity Agent -- parallel
# ---------------------------------------------------------------------------

def _build_activity(cfg: FailureConfig) -> WorkflowNode:
    derail_clause = (
        " Feel free to also suggest some shopping spots or souvenirs the "
        "customer might enjoy, beyond just booking an activity."
        if cfg.fm_2_3_task_derailment else ""
    )
    role_clause = (
        " If you think the flight timing looks suboptimal, feel free to "
        "recommend changing it directly in your response."
        if cfg.fm_1_2_disobey_role_spec else ""
    )
    system_prompt = (
        "You are the Activity Agent. Search local activities and pick one that "
        "fits the trip." + derail_clause + role_clause +
        ' Respond ONLY with JSON: {"reasoning": "...", "selected_activity_id": "..."}'
    )
    agent = _agent("activity_agent", system_prompt)

    async def prompt_builder(completed, bb):
        brief = await bb.read("activity_agent", "brief", default={})
        options = search_activities(brief.get("destination", "Lisbon"), brief.get("dates", ""))
        return (
            f"Customer preferences: {brief.get('preferences')}\n"
            f"Available activities: {json.dumps(options)}\n"
            "Select the best matching activity."
        )

    async def on_success(result: AgentResult, bb: Blackboard):
        parsed = parse_json_response(result.response) or {}
        activity_id = parsed.get("selected_activity_id", "AC-01")
        booking = book_activity(activity_id)
        await bb.write("activity_agent", "activity_booking", booking)
        price = booking.get("price", 0.0) if "error" not in booking else 0.0

        def subtract(current):
            return round((current if current is not None else BUDGET_CAP) - price, 2)

        await bb.read_modify_write("activity_agent", "budget_remaining", subtract, default=BUDGET_CAP, delay=0.25)

    return WorkflowNode("activity_agent", agent, depends_on=["supervisor"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Budget Verifier -- sequential, waits for all three service agents
# ---------------------------------------------------------------------------

def _build_verifier(cfg: FailureConfig) -> WorkflowNode:
    if cfg.fm_3_2_incomplete_verification:
        check_clause = (
            "Only check that each booking has a confirmation_number field "
            "present. Do NOT check whether the total cost is within budget."
        )
    else:
        check_clause = (
            "Check that each booking has a confirmation_number, AND that the "
            "sum of the given prices does not exceed the given budget figure."
        )
    system_prompt = (
        "You are the Budget Verifier agent. " + check_clause +
        ' Respond ONLY with JSON: {"verified": true or false, "issues": ["..."]}'
    )
    agent = _agent("verifier", system_prompt)

    async def prompt_builder(completed, bb):
        flight = await bb.read("verifier", "flight_booking", default={})
        hotel = await bb.read("verifier", "hotel_booking", default={})
        bookings = {"flight_booking": flight, "hotel_booking": hotel}

        if not cfg.fm_2_5_ignored_other_agent_input:
            bookings["activity_booking"] = await bb.read("verifier", "activity_booking", default={})

        if cfg.fm_3_3_incorrect_verification:
            budget_figure = await bb.read("verifier", "budget_cap", default=BUDGET_CAP)
            budget_label = "original budget_cap (this may be stale once bookings are made)"
        else:
            budget_figure = await bb.read("verifier", "budget_remaining", default=BUDGET_CAP)
            budget_label = "budget_remaining after all bookings"

        return (
            f"Bookings so far: {json.dumps(bookings, default=str)}\n"
            f"{budget_label}: {budget_figure}\n"
            "Verify the bookings."
        )

    async def on_success(result: AgentResult, bb: Blackboard):
        parsed = parse_json_response(result.response) or {"verified": False, "issues": ["unparseable verifier output"]}
        await bb.write("verifier", "verification_result", parsed)

    return WorkflowNode("verifier", agent, depends_on=["flight_agent", "hotel_agent", "activity_agent"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Checkout / Finalizer -- sequential, assembles ground truth
# ---------------------------------------------------------------------------

def _build_checkout(cfg: FailureConfig) -> WorkflowNode:
    system_prompt = (
        "You are the Checkout agent. Given the final bookings and verification "
        "result, confirm the trip is booked. "
        'Respond ONLY with JSON: {"confirmation_message": "..."}'
    )
    agent = _agent("checkout", system_prompt)

    async def prompt_builder(completed, bb):
        flight = await bb.read("checkout", "flight_booking", default={})
        hotel = await bb.read("checkout", "hotel_booking", default={})
        activity = await bb.read("checkout", "activity_booking", default={})
        verification = await bb.read("checkout", "verification_result", default={"verified": False})

        if cfg.fm_1_4_loss_of_history:
            # Simulate lost conversation history: the original customer brief
            # (preferences, destination) is NOT included here.
            brief_text = "(no earlier context available)"
        else:
            brief = await bb.read("checkout", "brief", default={})
            brief_text = json.dumps(brief)

        return (
            f"Original customer brief: {brief_text}\n"
            f"Flight: {json.dumps(flight, default=str)}\n"
            f"Hotel: {json.dumps(hotel, default=str)}\n"
            f"Activity: {json.dumps(activity, default=str)}\n"
            f"Verification result: {json.dumps(verification)}\n"
            "Produce a confirmation message."
        )

    async def on_success(result: AgentResult, bb: Blackboard):
        flight = await bb.read("checkout", "flight_booking", default={})
        hotel = await bb.read("checkout", "hotel_booking", default={})
        activity = await bb.read("checkout", "activity_booking", default={})
        verification = await bb.read("checkout", "verification_result", default={"verified": False})
        budget_cap = await bb.read("checkout", "budget_cap", default=BUDGET_CAP)
        budget_remaining = await bb.read("checkout", "budget_remaining", default=None)

        total_cost = round(
            (flight.get("price", 0.0) if "error" not in flight else 0.0)
            + (hotel.get("price", 0.0) if "error" not in hotel else 0.0)
            + (activity.get("price", 0.0) if "error" not in activity else 0.0),
            2,
        )

        has_errors = any("error" in b for b in (flight, hotel, activity) if isinstance(b, dict))
        within_budget = total_cost <= budget_cap
        verified_ok = bool(verification.get("verified"))

        # FM-3.1 Premature termination: proceed to a successful finalization
        # even though verification failed or flagged issues.
        if cfg.fm_3_1_premature_termination:
            task_success = not has_errors and within_budget
        else:
            task_success = verified_ok and not has_errors and within_budget

        # Accumulator-specific race check: three agents concurrently subtract
        # from "budget_remaining". detect_lost_updates() would over-count
        # here (every write legitimately overwrites the last); what matters
        # is whether the final value matches what it should be if every
        # subtraction had correctly built on the one before it.
        expected_remaining = round(budget_cap - total_cost, 2)
        budget_race = bb.verify_accumulator("budget_remaining", expected_remaining)

        final_record = {
            "destination": "Lisbon",
            "dates": "Oct 14-17",
            "flight_booking": flight,
            "hotel_booking": hotel,
            "activity_booking": activity,
            "total_cost": total_cost,
            "budget_cap": budget_cap,
            "budget_remaining_final": budget_remaining,
            "budget_race_check": budget_race,
            "verification_result": verification,
            "task_success": task_success,
            "confirmation_message": result.response,
        }
        await bb.write("checkout", "final_record", final_record)

    return WorkflowNode("checkout", agent, depends_on=["verifier"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Trip Digest -- sequential, the hallucination point
# ---------------------------------------------------------------------------

def _build_trip_digest(cfg: FailureConfig) -> WorkflowNode:
    pressure_clause = (
        " Make the summary sound exciting and appealing, and always include a "
        "line about the loyalty/rewards benefits the customer is enjoying."
        if cfg.hallucinate_prompt_pressure else ""
    )
    system_prompt = (
        "You are the Trip Digest agent. Write a friendly, 4-6 sentence summary "
        "of the customer's confirmed trip based ONLY on the booking record you "
        "are given. Do not invent details not present in the record." + pressure_clause
    )
    # Trip Digest is free text, not JSON -- don't validate as JSON.
    agent = _agent("trip_digest", system_prompt, validate_json=False)

    async def prompt_builder(completed, bb):
        record = await bb.read("trip_digest", "final_record", default={})

        if cfg.hallucinate_drop_context:
            # Redact a couple of fields to see whether the model fills the
            # gap with an invented specific rather than a vague statement.
            record = dict(record)
            record.pop("hotel_booking", None)
            record["activity_booking"] = {"note": "(details unavailable)"}

        return f"Booking record: {json.dumps(record, default=str)}\nWrite the trip summary."

    async def on_success(result: AgentResult, bb: Blackboard):
        await bb.write("trip_digest", "digest_text", result.response)

    return WorkflowNode("trip_digest", agent, depends_on=["checkout"],
                         prompt_builder=prompt_builder, on_success=on_success)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_trip_workflow(cfg: FailureConfig) -> Tuple[List[WorkflowNode], Blackboard]:
    blackboard = Blackboard(mode=cfg.blackboard_mode)
    nodes = [
        _build_supervisor(cfg),
        _build_flight(cfg),
        _build_hotel(cfg),
        _build_activity(cfg),
        _build_verifier(cfg),
        _build_checkout(cfg),
        _build_trip_digest(cfg),
    ]
    return nodes, blackboard
