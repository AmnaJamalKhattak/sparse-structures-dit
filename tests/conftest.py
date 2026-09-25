"""Suite-wide guards.

The most useful thing a test suite can do is fail for the reason it claims.
This file exists because one of these tests once did not: a smoke test asserted
that drawing prompts from an undersized cache raises, which was true only on a
machine with no route to DiffusionDB.  It passed in a sandbox and failed in
Colab, where the fetch it expected to fail instead succeeded -- and where it also
downloaded a dataset in the middle of a smoke test.

So the fetchers are severed for the whole suite.  A test that wants one stubs it
in itself; a test whose point *is* to reach DiffusionDB marks itself
``@pytest.mark.allows_network`` and says so out loud.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allows_network: this test deliberately reaches DiffusionDB. Expect it to be slow, "
        "and to fail when a third party is down rather than when the code is wrong.")


@pytest.fixture(autouse=True)
def offline(request, monkeypatch):
    """Cut the DiffusionDB fetchers unless the test asks for them by name.

    Only the prompt fetchers: no test in this suite loads real weights, so there
    is nothing else here that would leave the machine, and a broader patch would
    be guessing at seams rather than closing a known one.
    """
    if "allows_network" in request.keywords:
        return
    from ditsinks import prompt_sets

    def refuse(*args, **kwargs):
        raise AssertionError(
            "this test reached the network. Stub the fetch it needs, or mark it "
            "@pytest.mark.allows_network if reaching DiffusionDB is the point of the test.")

    monkeypatch.setattr(prompt_sets, "fetch_metadata_rows", refuse)
    monkeypatch.setattr(prompt_sets, "fetch_viewer_rows", refuse)
