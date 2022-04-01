from pathlib import Path

from dac4automlcomp.policy import DACPolicy

from baselines import schedulers


def load_solution(
    policy_cls=schedulers.ConstantPolicy, path=Path("tmp", "saved_configs")
) -> DACPolicy:
    """
    Load Solution.
    Serves as an entry point for the competition evaluation.
    By default (the submission) it loads a saved SMAC optimized configuration for the ConstantLRPolicy.
    Args:
        policy_cls: The DACPolicy class object to load
        path: Path pointing to the location the DACPolicy is stored
    Returns
    -------
    DACPolicy
    """
    return policy_cls.load(path)
