%a = 0.7;
%b = 0.8;

syms w I a b

zero = -b^3*w^3 + w^2*(2*b^2+a*b^2) - w*(b*a^2+2*a^2+3) - a^3 + I - 3*a;
coeffs = [-b^3, (2*b^2+a*b^2), (b*a^2+2*a^2+3), -a^3+I-3*a];

r = roots(coeffs);
simplify(r)
