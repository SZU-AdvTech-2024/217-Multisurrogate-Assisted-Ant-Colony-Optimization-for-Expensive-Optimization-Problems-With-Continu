function [v_r, v_o, v_c, f, cons, rank_out] = Initialize_cons(Par)
up_r = Par.up_r;
dn_r = Par.dn_r;
up_o = Par.up_o;
dn_o = Par.dn_o;
l = Par.l;
k = Par.k;
len_r = Par.len_r;
len_o = Par.len_o;
len_c = Par.len_c;
func_num = Par.func_num;

v_r = repmat((up_r - dn_r), k, 1) .* lhsdesign(k, len_r) + repmat(dn_r, k, 1);
v_o = repmat((up_o - dn_o), k, 1) .* lhsdesign(k, len_o) + repmat(dn_o, k, 1);
v_c = ceil(repmat(l, k, 1) .* lhsdesign(k, len_c));

% v_r = repmat((up_r - dn_r), k, 1) .* lhsdesign(k, len_r) + repmat(dn_r, k, 1);
% v_o = repmat((up_o - dn_o), k, 1) .* lhsdesign(k, len_o) + repmat(dn_o, k, 1);
% v_c = ceil(repmat((l - 1), k, 1) .* lhsdesign(k, len_c) + repmat(ones(1,len_c), k, 1));

% v_r = (up_r-dn_r).*rand(k,len_r) + dn_r;
% v_o = (up_o-dn_o).*rand(k,len_o) + dn_o;
% v_c = ceil(l.*rand(k,len_c));

% v_r(:,[1 3 4]) = 0.0625*ceil(v_r(:,[1 3 4])/0.0625);
[f, cons] = Design_Func(v_r, v_o, v_c,func_num);
CV = sum(max(0,cons),2);

idx_fea = (CV == 0);
if sum(idx_fea) == 0
    fmax = 0;
else
    fmax = max(f(idx_fea));
end
f_real(~idx_fea) = fmax + CV(~idx_fea);
f_real(idx_fea) = f(idx_fea);
[~,rank_v] = sort(f_real);
[~,rank_out] = sort(rank_v);