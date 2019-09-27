syms f(x) d
f(x) = (1/x) * exp(-d/x);
df = diff(f, x)
zero_df = solve(df==0, x)
pretty(df)
% syms g(x) t d
% g(x) = 1/(1-exp(-3/x)) * (1/x) * exp(-0.1/x);
% dg = diff(g, x)
% [solx, params, conditions] = solve(dg==0, x, 'ReturnConditions', true)

syms h(x) d
h(x) = -log((1/x) * exp(-d/x));
dh = diff(h, x)
zero_dh = solve(dh==0, x)