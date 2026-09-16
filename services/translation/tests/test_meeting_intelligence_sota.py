import pytest
from ramo_translate.intelligence import MeetingIntelligenceSynthesizer, TemporalDeduplicator, SemanticEvent


def test_sota_enterprise_schema_parsing():
    synth = MeetingIntelligenceSynthesizer()
    raw_llm_json = '''{
        "action_items": [
            {
                "task": "Deploy database migration",
                "assignee": "Alice",
                "assigned_by": "Bob",
                "deadline": "Friday 5pm",
                "dependencies": "Legal review",
                "urgency": "high"
            }
        ],
        "decisions": [
            {
                "decision": "Adopt PostgreSQL over MongoDB",
                "rationale": "ACID compliance for billing data",
                "rejected_alternatives": ["MongoDB", "DynamoDB"],
                "dissenting_views": "John noted concerns on horizontal scaling"
            }
        ],
        "schedule_dynamics": [
            {
                "event_type": "reschedule_sprint",
                "proposed_time": "Thursday 3:00 PM",
                "participants": ["Alice", "Bob", "Charlie"],
                "check_availability": true
            }
        ],
        "verification_claims": [
            {
                "claim": "Q3 churn dropped to 2.1%",
                "speaker": "Sarah",
                "metric_value": "2.1%"
            }
        ],
        "blockers_and_risks": [
            {
                "blocker": "Security audit not signed off",
                "affected_area": "Production deployment",
                "severity": "critical"
            }
        ],
        "unanswered_questions": [
            {
                "question": "Who has access to the prod AWS credentials?",
                "asked_by": "Dave",
                "target_person": "Ops team"
            }
        ],
        "key_notes": [
            "Team aligned on Q4 timeline"
        ],
        "events": [
            {
                "event_id": "evt-12345",
                "event_category": "ACTION_ITEM",
                "confidence_score": 0.92,
                "source_speakers": ["Bob", "Alice"],
                "trigger_quote": "Alice, please deploy the database migration by Friday 5pm.",
                "structured_payload": {
                    "task": "Deploy database migration",
                    "assignee": "Alice",
                    "deadline": "Friday 5pm"
                }
            },
            {
                "event_id": "evt-67890",
                "event_category": "BLOCKER_RISK",
                "confidence_score": 0.88,
                "source_speakers": ["Alice"],
                "trigger_quote": "We can't deploy because the security audit is not signed off.",
                "structured_payload": {
                    "blocker": "Security audit not signed off",
                    "severity": "critical"
                }
            }
        ]
    }'''

    parsed = synth.parse_output(raw_llm_json)

    # 1. Backward compatibility keys
    assert len(parsed["action_items"]) == 1
    assert parsed["action_items"][0]["urgency"] == "high"
    assert parsed["action_items"][0]["dependencies"] == "Legal review"

    # 2. SOTA enterprise categories
    assert len(parsed["decisions"]) == 1
    assert parsed["decisions"][0]["decision"] == "Adopt PostgreSQL over MongoDB"
    assert "MongoDB" in parsed["decisions"][0]["rejected_alternatives"]

    assert len(parsed["schedule_dynamics"]) == 1
    assert parsed["schedule_dynamics"][0]["proposed_time"] == "Thursday 3:00 PM"
    assert parsed["schedule_dynamics"][0]["check_availability"] is True

    assert len(parsed["verification_claims"]) == 1
    assert parsed["verification_claims"][0]["metric_value"] == "2.1%"

    assert len(parsed["blockers_and_risks"]) == 1
    assert parsed["blockers_and_risks"][0]["severity"] == "critical"

    assert len(parsed["unanswered_questions"]) == 1
    assert parsed["unanswered_questions"][0]["asked_by"] == "Dave"

    assert len(parsed["key_notes"]) == 1

    # 3. Standardized SemanticEvent objects
    assert len(parsed["events"]) == 2
    evt1 = parsed["events"][0]
    assert evt1.event_category == "ACTION_ITEM"
    assert evt1.confidence_score == 0.92
    assert evt1.requires_commander_routing is True  # > 0.85 threshold


def test_temporal_deduplication():
    dedup = TemporalDeduplicator(similarity_threshold=0.85)

    evt1 = SemanticEvent(
        event_id="e1",
        event_category="ACTION_ITEM",
        confidence_score=0.90,
        source_speakers=["Bob"],
        trigger_quote="Alice will deploy the database migration by Friday.",
        structured_payload={"task": "Deploy database migration", "assignee": "Alice", "deadline": "Friday"},
    )

    # First event is an INSERT
    res1 = dedup.process_event(evt1)
    assert res1.is_update is False
    assert dedup.active_event_count == 1

    # Second event across next turn with similar intent is an UPDATE
    evt2 = SemanticEvent(
        event_id="e2",
        event_category="ACTION_ITEM",
        confidence_score=0.94,
        source_speakers=["Bob", "Alice"],
        trigger_quote="Alice confirmed she will deploy the database migration by Friday afternoon.",
        structured_payload={"task": "Deploy database migration", "assignee": "Alice", "deadline": "Friday 2pm"},
    )

    res2 = dedup.process_event(evt2)
    assert res2.is_update is True
    assert res2.canonical_event_id == "e1"
    assert dedup.active_event_count == 1
    # Deadline updated to more specific constraint
    assert dedup.get_event("e1").structured_payload["deadline"] == "Friday 2pm"
