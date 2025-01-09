function out = VCH(s,sl)
[~,n] = size(s);
for i = 1:n
    ds = sqrt( sum( (sl(:,i:n) - s(:,i:n)).^2, 2) );
    p  = ds.^4 / sum(ds.^4);
    idx = Select(p);
    A(i,:) = sl - s(idx,:);
    s(idx,:) = [];
end
if max(max(A))<1e-5
    B = Gram_Schmidt_process(rand(n));
else
    B = Gram_Schmidt_process(A');
end
out = B;