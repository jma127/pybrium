import torch

from pybrium import nash_average


def rock_paper_scissors():
    print("==== Rock paper scissors ====")

    payoffs = torch.tensor([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=torch.float32)

    print("Payoffs:")
    print(payoffs)

    nash_equilibrium, nash_rating = nash_average(payoffs, steps=(2 ** 14))

    print("Equilibrium strategy:")
    print(nash_equilibrium)

    print("Nash averaging ratings:")
    print(nash_rating)


def redundancy_invariance():
    print("==== Rock paper scissors knife ====")

    payoffs = torch.tensor([[0, 1, -1, -1], [-1, 0, 1, 1], [1, -1, 0, 0], [1, -1, 0, 0]], dtype=torch.float32)

    print("Payoffs:")
    print(payoffs)

    nash_equilibrium, nash_rating = nash_average(payoffs, steps=(2 ** 14))

    print("Equilibrium strategy:")
    print(nash_equilibrium)

    print("Nash averaging ratings:")
    print(nash_rating)


def random_tournament():
    print("==== Random tournaments ====")

    n = 8

    logit_winrates = torch.randn((n, n))
    logit_winrates = logit_winrates - logit_winrates.transpose(0, 1)

    print("Winrate logits:")
    print(logit_winrates)

    nash_equilibrium, nash_rating = nash_average(logit_winrates, steps=(2 ** 18))

    print("Equilibrium strategy:")
    print(nash_equilibrium)

    print("Nash averaging ratings:")
    print(nash_rating)


def main():
    rock_paper_scissors()
    redundancy_invariance()
    random_tournament()


if __name__ == "__main__":
    main()
