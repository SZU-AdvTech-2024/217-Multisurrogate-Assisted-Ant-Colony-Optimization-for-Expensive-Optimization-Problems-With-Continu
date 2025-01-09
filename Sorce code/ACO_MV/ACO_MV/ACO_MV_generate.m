function [v_r_generate,v_o_generate,v_c_generate] = ACO_MV_generate(v_r,v_c,len_r,len_c,k,kesi,l,q,rank_v,m)
global up_r dn_r
w = ( 1/(q*k*sqrt(2*pi)) )*exp( -((rank_v-1).^2)/(2*(q*k)^2) );
p_ro = w/sum(w);
for i = 1:m
    %生成连续部分
    v_r_generate(i,:) = ACO_RO(v_r,p_ro,len_r,k,kesi);
    v_r_generate(i,:) = Repair(v_r_generate(i,:),up_r,dn_r);
    
    %生成序数部分
    v_o_generate(i,:) = 0;
        
    %生成类别部分
    v_c_generate(i,:) = ACO_C(v_c,l,len_c,q,w);
end