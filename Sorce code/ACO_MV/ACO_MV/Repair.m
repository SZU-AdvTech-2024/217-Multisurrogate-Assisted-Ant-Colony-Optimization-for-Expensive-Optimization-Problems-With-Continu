function out = Repair(x,up,dn)
x = (x>=up).*max(dn,2*up-x) + (x<up).*x;
x = (x<=dn).*min(up,2*dn-x) + (x>dn).*x;
out = x;