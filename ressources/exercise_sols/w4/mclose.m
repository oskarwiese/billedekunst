function Io = mclose(I, se)

It = imdilate(I, se);
Io = imerode(It, se);
