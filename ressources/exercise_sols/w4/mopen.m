function Io = mopen(I, se)

It = imerode(I, se);
Io = imdilate(It, se);
