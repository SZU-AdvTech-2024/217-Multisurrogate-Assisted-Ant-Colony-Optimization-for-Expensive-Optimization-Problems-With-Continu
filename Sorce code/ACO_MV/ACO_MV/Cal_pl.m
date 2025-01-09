function [out] = Cal_pl(vc,l,w,q)

for i = 1:l
    idx_l = ( vc==i );
    u(i) = sum( idx_l );

    if isempty(w(idx_l))
       wjl(i) = 0; 
    else
       wjl(i) = max(w(idx_l));
    end
end
eta = 100*sum( u==0 );
for i = 1:l
    if (eta>0)&&(u(i)>0)
        wl(i) = wjl(i)/u(i) + q/eta;
    elseif (eta==0)&&(u(i)>0)
        wl(i) = wjl(i)/u(i);
    elseif (eta>0)&&(u(i)==0)
        wl(i) = q/eta;
    end
end

% for i = 1:l
%     if (eta>0)&&(u(i)>0)
%         wl(i) = wjl(i)/u(i) + q/eta;
%     elseif (eta==0)&&(u(i)>0)
%         wl(i) = wjl(i)/u(i);
%     elseif (eta>0)&&(u(i)==0)
%         wl(i) = q/eta;
%     end
% end
out = wl/sum(wl);