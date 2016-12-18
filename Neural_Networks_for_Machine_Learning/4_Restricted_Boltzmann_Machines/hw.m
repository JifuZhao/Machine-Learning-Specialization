clear all;
a4_init;

%%
a4_main(300, 0, 0, 0);

%%
describe_matrix(visible_state_to_hidden_probabilities(test_rbm_w, data_37_cases));

%%
describe_matrix(hidden_state_to_visible_probabilities(test_rbm_w, test_hidden_state_37_cases));

%%
configuration_goodness(test_rbm_w, data_37_cases, test_hidden_state_37_cases)

%%
describe_matrix(configuration_goodness_gradient(data_37_cases, test_hidden_state_37_cases));

%%
describe_matrix(cd1(test_rbm_w, data_37_cases));

%%
describe_matrix(cd1(test_rbm_w, data_37_cases));

%%
a4_main(300, .02, .1, 1000);