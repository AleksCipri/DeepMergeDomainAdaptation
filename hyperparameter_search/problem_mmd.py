from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_dim("lr", [1e-4, 5e-4, .001, .005, .01, .1])
Problem.add_dim("trade_off", [.01, .05, .1, .5, 1])
Problem.add_dim("cycle_length", [2, 4, 8, 10])
Problem.add_dim("weight_decay", [1e-5, 1e-4, 1e-3, 5e-2, .01, .1])

Problem.add_starting_point(lr=1e-4, trade_off=.01, cycle_length=2, weight_decay = 1e-5)

if __name__ == "__main__":
    print(Problem)
