import torch

from .maxent_ce import solve_maxent_ce


def nash_average(payoffs, steps=1000000, tol=1e-6, lams=None, lr=None):
    """Calculates the mixed Nash equilibrium strategy and the resulting
    Nash average, as described by Balduzzi et al. (2018).

    Args:
        payoffs (torch.Tensor):
            Antisymmetric payoff matrix.
        steps (int, optional):
            Number of SGD steps to use in calculations (default: 1000000).
        tol (float, optional):
            Tolerance for asymmetries (default: 1e-6).
        lams (torch.Tensor):
            Initialization logits (default: auto-initialied).
        lr (float):
            SGD learning rate (default: auto-computed).
    Returns:
        Tuple[torch.Tensor, torch.Tensor] whose first element is the mixed Nash
        equilibrium strategy with maximum entropy, and whose second element is
        the Nash averaging skill ratings.

    Balduzzi et al., "Re-evaluating Evaluation", 2018,
        https://arxiv.org/abs/1806.02643
    """
    assert payoffs.dim() == 2 and payoffs.size(0) == payoffs.size(1)

    payoff_stack = torch.stack([payoffs, -payoffs])

    solution = solve_maxent_ce(payoff_stack, steps=steps, lams=lams, lr=lr)
    reduced_strategy = solution.sum(dim=1)
    nash_avg = torch.mm(payoffs, reduced_strategy.view(-1, 1))

    assert (torch.abs(solution - solution.transpose(0, 1)) < tol).all()

    return reduced_strategy, nash_avg
