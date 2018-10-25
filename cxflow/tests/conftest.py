import pytest


@pytest.fixture
def anchorless_yaml():
    yield """
          e:
            f: f
            h:
              - j
              - k
          """


@pytest.fixture
def anchored_yaml():
    yield """
          a: &anchor
            b: c
            d: 11
        
          e:
            <<: *anchor
            f: f
            h:
              - j
              - k
          """
