function [out] = ACO_RO(x,p,len,k,kesi)
global zzz BBB v000 xxxxx idx_grrr
idx_gr = Select(p);
[z,B,v0] = Rot(x,idx_gr,len);
% z = x;
idx_grrr = idx_gr;
zzz = z;
BBB = B;
v000 = v0;
xxxxx = x;
for j = 1:len
    mu = z(idx_gr,j);
    sigma = kesi*sum( abs( z(idx_gr,j)-z(:,j) ) )/(k-1);
           
    n_z(1,j) = mu + sigma*randn(1);     
end
out = n_z*B'+v0;

function [z_r,B,v0] = Rot(v_r,idx_gr,len_r)
if (sum(sum(v_r - v_r(idx_gr,:))) ~= 0)&&(len_r>1)
    B = VCH(v_r,v_r(idx_gr,:));
else
    B = eye(len_r);
end

if rank(B) ~= len_r
%     B = B;
% else
    B = eye(len_r);
end

z_r = (v_r - v_r(idx_gr,:))*B;
% z_r = v_r*B;
v0 = v_r(idx_gr,:);