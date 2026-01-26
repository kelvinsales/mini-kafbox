% 1. Gerar dados de exemplo
n_samples = 500;
X = linspace(-10, 10, n_samples)'; % 500 x 1
Y = sin(X) + 0.1 * randn(n_samples, 1); % 500 x 1, seno com ruído

% 2. Definir a arquitetura e os hiperparâmetros
arq_rede = [100 10]; % Duas camadas ocultas
hiper_par = struct("epocas", 500,...
                  "tx_ap", 0.005,...
                  "tam_batch", 3
                  );   % Dados simulação

% 3. Treinar o modelo1
clc;
model = mlp_K_Treino(X, Y, arq_rede, hiper_par);

% 4. Fazer predições com novos dados
X_test = linspace(-12, 12, 100)';
Y_pred = mlp_K_Pred(X_test, model);

% 5. Visualizar os resultados
figure;
plot(X, Y, 'bo'); % Dados de treinamento
hold on;
plot(X_test, Y_pred, 'r-', 'LineWidth', 2); % Previsão do modelo
legend('Dados Verdadeiros', 'Previsão MLP');
title('Treinamento e Predição de MLP para Regressão');
grid on;
