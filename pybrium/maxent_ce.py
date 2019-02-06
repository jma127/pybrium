import torch


def _lams_to_log_policy(lams, payoffs):
    n = payoffs.size(0)
    action_counts = tuple(payoffs.shape[1:])
    log_policy = payoffs.new_zeros(action_counts)

    for i in range(n):
        lam = lams[i]
        ac = action_counts[i]
        payoff = payoffs[i]
        ohsize = [1] * n
        ohsize[i] = ac
        ohsize = tuple(ohsize)
        payoff_permuted = payoff.transpose(0, i)
        permuted_shape = payoff_permuted.shape

        assert lam.shape == (ac, ac)

        gain_mat = payoff_permuted.view(ac, 1, -1) - payoff_permuted.view(1, ac, -1)
        sub = (lam.transpose(0, 1).view(ac, ac, 1) * gain_mat).sum(dim=0).view(permuted_shape).transpose(0, i)
        log_policy.sub_(sub)

    return torch.nn.functional.log_softmax(log_policy.view(-1), dim=0).view(action_counts)


def _get_regret(policy, payoffs, positive=True):
    n = payoffs.size(0)
    action_counts = tuple(payoffs.shape[1:])

    ret = []
    for i in range(n):
        ac = action_counts[i]
        payoff = payoffs[i]

        policy_permuted = policy.transpose(0, i)
        payoff_permuted = payoff.transpose(0, i)

        gain_mat = payoff_permuted.view(ac, 1, -1) - payoff_permuted.view(1, ac, -1)
        r_mat = gain_mat.transpose(0, 1) * policy_permuted.view(ac, 1, -1)
        if not positive:
            r_mat = -r_mat

        r_mat = torch.nn.functional.relu(r_mat)
        ret.append(r_mat.view(ac, ac, -1).sum(dim=2))

    return ret


def solve_maxent_ce(payoffs, steps=1000000, lams=None, lr=None):
    """Calculates the maximum-entropy correlated equilibrium as defined in
    Ortiz et al. (2007).

    payoffs (torch.Tensor):
        Joint payoff tensor.
    steps (int, optional):
        Number of SGD steps to use in calculations (default: 1000000).
    lams (torch.Tensor):
        Initialization logits (default: auto-initialied).
    lr (float):
        SGD learning rate (default: auto-computed).

    Ortiz et al., "Maximum entropy correlated equilibria", 2007,
        http://proceedings.mlr.press/v2/ortiz07a/ortiz07a.pdf
    """
    n = payoffs.size(0)
    action_counts = tuple(payoffs.shape[1:])

    if lr is None:
        tot = 0.0
        for i in range(n):
            ac = action_counts[i]
            payoff_permuted = payoffs[i].transpose(0, i)
            gain_mat = payoff_permuted.view(ac, 1, -1) - payoff_permuted.view(1, ac, -1)
            tot += torch.abs(gain_mat).sum(dim=0).max().item()
        lr = 0.9 / tot

    if lams is None:
        lams = [(lr * payoffs.new_ones((i, i))) for i in action_counts]
        for i in range(n):
            rac = torch.arange(action_counts[i])
            lams[i][rac, rac] = 0.0

    for _ in range(steps):
        log_policy = _lams_to_log_policy(lams, payoffs)
        policy = torch.exp(log_policy)

        pos_regrets = _get_regret(policy, payoffs, positive=True)
        neg_regrets = _get_regret(policy, payoffs, positive=False)

        eps = 0.5 ** 125

        for i in range(n):
            ac = action_counts[i]
            rac = torch.arange(ac)

            chg = ((pos_regrets[i] + eps) / (pos_regrets[i] + neg_regrets[i] + 2 * eps)) - 0.5
            chg[rac, rac] = 0.0
            lams[i].add_(lr, chg)
            lams[i].clamp_(min=0.0)

    return policy
