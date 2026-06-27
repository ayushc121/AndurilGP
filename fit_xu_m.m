u0    = [8.1; 11.2; 13.0];
Xu_eff = [-0.413; -0.559; -0.656];
% Fit: Xu_eff = Xu_m_true + Xuu_m*u0
Phi = [ones(3,1), u0];
z   = Xu_eff;
p   = lsqlin(Phi, z, [], [], [], [], [-Inf; -Inf], [-1e-4; -1e-4]);
Xu_m_true = p(1);
Xuu_m     = p(2);

result = Xu_m_true.*ones(3,1) + Xuu_m.*u0;