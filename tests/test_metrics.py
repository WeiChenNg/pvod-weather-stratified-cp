import numpy as np
import pandas as pd

from src.metrics import picp, pinaw, summarize, winkler

Y = np.array([0.5, 0.1, 0.9])
LO = np.array([0.4, 0.2, 0.4])
HI = np.array([0.6, 0.3, 0.8])


def test_hand_computed():
    assert np.isclose(picp(Y, LO, HI), 1 / 3)
    assert np.isclose(pinaw(LO, HI), 0.7 / 3)
    # 0.2；0.1 + 2/0.1*0.1；0.4 + 2/0.1*0.1
    assert np.isclose(winkler(Y, LO, HI, alpha=0.1), (0.2 + 2.1 + 2.4) / 3)


def test_summarize_overall_and_by_class():
    r = pd.DataFrame({"y": np.r_[Y, np.nan], "lo": np.r_[LO, 0], "hi": np.r_[HI, 1],
                      "wclass": ["clear", "clear", "cloudy", "cloudy"]})
    s = summarize(r, alpha=0.1).set_index("wclass")
    assert s.loc["all", "n"] == 3
    assert np.isclose(s.loc["clear", "picp"], 0.5)
    assert np.isclose(s.loc["cloudy", "picp"], 0.0)


def test_summarize_nan_wclass_counts_in_all_only():
    r = pd.DataFrame({"y": [0.5, 0.5, 0.5], "lo": [0.4, 0.4, 0.6], "hi": [0.6, 0.6, 0.7],
                      "wclass": ["clear", np.nan, "cloudy"]})
    s = summarize(r, alpha=0.1).set_index("wclass")
    assert s.loc["all", "n"] == 3
    assert set(s.index) == {"all", "clear", "cloudy"}
    assert s.loc["clear", "n"] == 1
    assert s.loc["cloudy", "n"] == 1
    assert np.isclose(s.loc["all", "picp"], 2 / 3)
