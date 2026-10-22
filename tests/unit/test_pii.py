from src.metadata import pii


def test_email_and_phone_masked():
    text = "Reach me at priya.raman@example.com or +1 555 0101 or (512) 555-0147."
    out = pii.mask_pii(text)
    assert out == "Reach me at [EMAIL] or [PHONE] or [PHONE]."


def test_urls_masked():
    out = pii.mask_pii("linkedin.com/in/priya-raman, github.com/priyaraman, https://priya.dev/cv")
    assert out == "[URL], [URL], [URL]"


def test_name_masked_in_full_and_in_parts_case_insensitively():
    out = pii.mask_pii("Priya Raman led the team. raman also wrote the API; PRIYA agreed.",
                       names=["Priya Raman"])
    assert out == "[NAME] led the team. [NAME] also wrote the API; [NAME] agreed."


def test_name_inside_other_words_is_left_alone():
    assert pii.mask_pii("Ramanujan and Priyanka", names=["Priya Raman"]) == "Ramanujan and Priyanka"


def test_dates_and_numbers_are_not_phones():
    text = "Mar 2021 - Dec 2024, 3,000 requests per second, 2018 - 2021"
    assert pii.mask_pii(text) == text


def test_no_pii_left_in_a_typical_chunk():
    chunk = "Priya Raman | priya.raman@example.com | +1 555 0101 | Austin, TX\nSenior Engineer"
    out = pii.mask_pii(chunk, names=["Priya Raman"])
    for secret in ("Priya", "Raman", "@", "555 0101"):
        assert secret not in out
    assert "Senior Engineer" in out and "Austin, TX" in out


def test_email_hash_is_stable_and_case_insensitive():
    a = pii.email_hash("Priya.Raman@Example.com ")
    assert a == pii.email_hash("priya.raman@example.com")
    assert len(a) == 64 and "priya" not in a
    assert pii.email_hash("") == ""
