import torch

from pybrium.nash_averaging import nash_average


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


def main():
    rock_paper_scissors()
    redundancy_invariance()


if __name__ == "__main__":
    main()
