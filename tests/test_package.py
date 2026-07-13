import predictive_maintenance


def test_package_exposes_version() -> None:
    assert isinstance(predictive_maintenance.__version__, str)
    assert predictive_maintenance.__version__
