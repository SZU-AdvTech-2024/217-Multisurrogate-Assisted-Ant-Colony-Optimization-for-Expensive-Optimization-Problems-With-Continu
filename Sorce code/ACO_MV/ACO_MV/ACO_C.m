function [n_x] = ACO_C(x,l,len_c,q,w)
for j = 1:len_c
    [pl] = Cal_pl(x(:,j),l(:,j),w,q);
    idx_gc = Select(pl);            

    n_x(1,j) = idx_gc;
end 