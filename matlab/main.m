n = 100;
m = 1000;
k = 300;
mix_step = 2000;
num_run = 100;

X = randn(n, m);

rst_rand = 0;
rst_dvs = 0;

for i = 1:num_run
    C = randperm(m, k);
    curr_rst_rand = norm(pinv(X(:,C) * X(:,C)'), 'fro');
    rst_rand = rst_rand + curr_rst_rand;

    C = dvsMc(X, mix_step, k);
    curr_rst_dvs = norm(pinv(X(:,C) * X(:,C)'), 'fro');
    rst_dvs = rst_dvs + curr_rst_dvs;
end

rst_rand / num_run
rst_dvs / num_run


