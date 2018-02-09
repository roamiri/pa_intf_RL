function R = rate_mesh(G, P1, P2, p3 , Beta, sigma2, Gamma)
    
    gamma_1 = 0;
    s1 = G(1).*P1;
    inf1 = G(1).*P2.*Beta(2,1);
    inf1 = inf1 + G(1)*p3*Beta(3,1)+sigma2;
    gamma_1 = s1./inf1;
    
    gamma_2 = 0.0;
    s2 = G(2).*P2;
    inf2 = G(2).*P1.*Beta(1,2);
    inf2 = inf2 + G(2)*p3*Beta(3,2)+sigma2;
    gamma_2 = s2./inf2;
    
    gamma_3 = 0.0;
    s3 = G(3).*P2;
    inf3 = G(3).*P1.*Beta(1,3);
    inf3 = inf3 + G(3).*P2.*Beta(2,3)+sigma2;
    gamma_3 = s3./inf3;
    
    r_1 = log2(1+gamma_1./Gamma);
    r_2 = log2(1+gamma_2./Gamma);
    r_3 = log2(1+gamma_3./Gamma);
    
    R = r_1 + r_2 + r_3;
end