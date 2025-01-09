function i = Select(p)
p_sel = cumsum(p);
R     = rand;
i = 1;
while p_sel(i) < R
    i = i+1;
end