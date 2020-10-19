from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_dim("lr", [1e-4, 5e-4, .001, .005, .01, .1])
Problem.add_dim("trade_off", [.01, .05, .1, .5, 1])
Problem.add_dim("intra_loss_coef", [.01, .05, .1, .5, 1])
Problem.add_dim("inter_loss_coef", [.01, .05, .1, .5, 1])
Problem.add_dim("em_loss_coef", [.01, .05, .1, .5, 1])
Problem.add_dim("cycle_length", [2, 4, 8, 10])
Problem.add_dim("weight_decay", [1e-4, 1e-3, 5e-2, .01, .1])
Problem.add_dim("ad_net_mult_lr", [.01, .05, .1, .5, .75, 1.1, 1.5])
#Problem.add_dim("beta_1", [.7, .8, .9])
#Problem.add_dim("beta_2", [.8, .9, .99])

Problem.add_starting_point(lr=1e-4, trade_off=.01, intra_loss_coef=.01, inter_loss_coef=.01, em_loss_coef=.01, cycle_length=2, \
	weight_decay = 1e-4, ad_net_mult_lr = .01, beta_1 = .7, beta_2 = .8)

if __name__ == "__main__":
    print(Problem)