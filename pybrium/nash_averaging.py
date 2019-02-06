import torch

from .maxent_ce import solve_maxent_ce


def nash_average(payoffs, steps=1000000, tol=1e-6, lams=None, lr=None):
    assert payoffs.dim() == 2 and payoffs.size(0) == payoffs.size(1)

    payoff_stack = torch.stack([payoffs, -payoffs])

    solution = solve_maxent_ce(payoff_stack, steps=steps, lams=lams, lr=lr)
    reduced_policy = solution.sum(dim=1)
    nash_avg = torch.mm(payoffs, reduced_policy.view(-1, 1))

    assert (torch.abs(solution - solution.transpose(0, 1)) < tol).all()

    return reduced_policy, nash_avg
