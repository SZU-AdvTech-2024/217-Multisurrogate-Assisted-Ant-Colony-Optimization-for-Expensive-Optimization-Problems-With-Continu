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

load('E:\Matlab_2017a\bin\Surrogate\Surrogate_MVOP_2019_BX\ProcessData\Design_Pop')

v_r = Arc_design(1:60,1:5);
v_o = repmat((up_o - dn_o), k, 1) .* lhsdesign(k, len_o) + repmat(dn_o, k, 1);
v_c = Arc_design(1:60,6:10);

f = Arc_design_f(1:60,1);
cons = Arc_design_f(1:60,2:end);

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