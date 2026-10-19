from complaints.agent import policies, triage

QUEUES = ["Billing dispute", "Fees and interest", "Other"]


def case():
    return triage.Case(1, "I was charged twice for one purchase.", "Store credit card", "Bank",
                       {"Billing dispute": 0.7, "Fees and interest": 0.2, "Other": 0.1}, 0.62,
                       [{"issue": "Problem with a purchase", "response": "relief", "text": "Double charge"}])


def test_policies_parse():
    p = policies.load_policies()
    assert len(p) == 12 and "POL-01" in p
    assert "Billing dispute" in p["POL-01"].queues
    assert {x.id for x in policies.policies_for("Payments")} >= {"POL-05", "POL-06", "POL-11"}


def test_messages_include_tools_only_when_asked():
    with_tools = triage.messages(case(), QUEUES, True)[1]["content"]
    without = triage.messages(case(), QUEUES, False)[1]["content"]
    assert "predict_relief() -> probability the complaint ends in relief: 0.62" in with_tools
    assert "POL-01" in with_tools and "Tool" not in without
    assert triage.provided_policies(case(), True) >= {"POL-01", "POL-04", "POL-11"}


def test_parse_and_score():
    good = ('Here: {"queue": "Billing dispute", "priority": "high", "cited_policies": ["POL-01"], '
            '"reply": "Thank you for telling us about the duplicate charge. We have opened a review."}')
    unsafe = '{"queue": "Other", "priority": "low", "cited_policies": ["POL-99"], "reply": "We will refund it."}'
    rows = [{"true_queue": "Billing dispute", "relief": 1, "provided": {"POL-01"}, "raw": good},
            {"true_queue": "Billing dispute", "relief": 0, "provided": {"POL-01"}, "raw": unsafe},
            {"true_queue": "Other", "relief": 0, "provided": set(), "raw": "not json"}]
    s = triage.score(rows, QUEUES)
    assert s["json_valid"] == 2 / 3 and s["queue_accuracy"] == 0.5
    assert s["citations_exist"] == 0.5 and s["citations_grounded"] == 0.5
    assert s["unsafe_promise_rate"] == 0.5
    assert s["priority"]["high"]["relief_rate"] == 1.0


def test_unsafe_pattern():
    assert triage.UNSAFE.search("Your fee will be waived today")
    assert triage.UNSAFE.search("We'll reverse the charge")
    assert not triage.UNSAFE.search("We will review the charge and reply within 15 days")


def test_guard_keeps_queue_in_router_top3_and_sets_priority():
    raw = '{"queue": "Payments", "escalate": false, "cited_policies": ["POL-01"], "reply": "Thanks."}'
    final, flags = triage.guard(raw, case(), high=0.5, medium=0.3)
    out = triage.parse(final)
    assert flags["queue_fallback"] and out["queue"] == "Billing dispute"  # Payments is not in the top 3
    assert out["priority"] == "high"                                      # relief_p 0.62 >= 0.5
    raw = '{"queue": "Fees and interest", "escalate": true, "cited_policies": [], "reply": "Thanks."}'
    final, flags = triage.guard(raw, case(), high=0.9, medium=0.7)
    assert triage.parse(final)["queue"] == "Fees and interest" and flags.get("escalated")
    assert triage.parse(final)["priority"] == "high"
