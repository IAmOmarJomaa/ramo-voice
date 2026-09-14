import pytest
from ramo_translate.fast_bypass import check_fast_bypass, FAST_BYPASS_DICT


def test_fast_bypass_french():
    assert check_fast_bypass("Okay.", "French") == "D'accord."
    assert check_fast_bypass("yes", "fr") == "Oui."
    assert check_fast_bypass("No!", "French") == "Non."
    assert check_fast_bypass("Thank you", "fr") == "Merci !"
    assert check_fast_bypass("Sounds good.", "French") == "Ça marche."


def test_fast_bypass_spanish():
    assert check_fast_bypass("Okay.", "Spanish") == "De acuerdo."
    assert check_fast_bypass("yes", "es") == "Sí."
    assert check_fast_bypass("thank you!", "es") == "¡Gracias!"
    assert check_fast_bypass("goodbye", "Spanish") == "Adiós."


def test_fast_bypass_german_and_arabic():
    assert check_fast_bypass("yes", "German") == "Ja."
    assert check_fast_bypass("no", "de") == "Nein."
    assert check_fast_bypass("thanks", "Arabic") == "شكراً."
    assert check_fast_bypass("yes", "ar") == "نعم."


def test_fast_bypass_miss_for_complex_sentences():
    # Long/complex sentences must return None so they route to LLM
    assert check_fast_bypass("We need to complete the migration by Friday.", "French") is None
    assert check_fast_bypass("What is the status of the database?", "Spanish") is None
