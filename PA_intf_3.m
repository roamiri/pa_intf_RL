clc;
clear;
%%
p1_max = 10;
p2_max = 20;
p3_max = 20;

p1 = linspace(0,p1_max, 100);
p2 = linspace(0,p2_max, 100);
p3 = linspace(0,p3_max, 10);

G = [1.5 2.5 2.0];

Beta = [1 0.1 0.1; 0.1 1 0.1; 0.1 0.1 1];

sigma2 = 1;
Gamma = 3.532;
%%
[X,Y] = meshgrid(p1, p2);
figure;
hold on;
grid on;
box on;

p3 = 0.0;
for i=1:numel(p3)
   r1 = rate_mesh(G, X, Y, p3(i) , Beta, sigma2, Gamma); 
   surf(X, Y, r1)
end
colorbar
