

function model = kn_K_Treino(X, Y, options)
% Implementacao KNLMS baseado no Artigo de Cedric 2009/KAF box
%
% Objetivo: Aplicacao da knlms em predicao de chuvas
%
% Autor: Allan Kelvin M Sales
% Data: 24/11/2025
%
% Treina um modelo KNLMS.
% Input:
%   X: n x d matriz de dados de treinamento.
%   Y: n x 1 vetor de rótulos de treinamento.
%   options: estrutura com os hiperparâmetros:
%       .limiar de coerencia (mu0)
%       .taxa de aprendizagem (eta).
%       .regularizacao (reg)
%       .kernel ('gauss', etc.)
%       .kernel_gamma (gamma)
%
% Output:
%   model: estrutura contendo o modelo treinado:
%       .dict: o dicionário de vetores de suporte.
%       .alpha: os coeficientes do modelo (akn).
%       .options: os hiperparâmetros usados no treino.

    % --- 1. Extrair Hiperparâmetros ---
    [mu0, eta, reg] = options.hiperpar{:};
    [k_tipo, k_gamma] = options.kernel{:};

    Nx = size(X, 1);

    % --- 2. Inicializar o Modelo ---
    dict   = [];  % Dicionário (vetores de suporte)
    modict = [];  % Módulo L2 dos vetores do dicionário
    akn    = [];  % Coeficientes do modelo

    fprintf('Iniciando o treinamento KNLMS para %d amostras...\n', Nx);
    tic;

    % --- 3. Loop de Treinamento Online ---
    for t = 1:Nx
        x_current = X(t, :);
        y_current = Y(t, :);

        if isempty(dict)
            % Primeira amostra: inicia o dicionário
            dict = x_current;
            kx = kernel_K(x_current, x_current, k_tipo, k_gamma);
            modict = sqrt(kx);
            akn = 0;
        else
            % Critério de Coerência para decidir se adiciona ao dicionário
            kf = kernel_K(dict, x_current, k_tipo, k_gamma);
            kx = kernel_K(x_current, x_current, k_tipo, k_gamma);

            coherence = max( abs(kf ./ (sqrt(kx) * modict)) );

            if coherence <= mu0
                % Amostra é "nova", aumenta o dicionário
                dict   = [dict; x_current];
                modict = [modict; sqrt(kx)];
                akn    = [akn; 0];
            end
        end

        % Atualiza os coeficientes usando a regra do KNLMS
        kf = kernel_K(dict, x_current, k_tipo, k_gamma);
        pred_error = y_current - kf' * akn;

        akn = akn + (eta / (reg + kf' * kf)) * pred_error * kf;
    end

    training_time = toc;
    fprintf('Treinamento concluído em %.2f segundos.\n', training_time);
    fprintf('Tamanho final do dicionário: %d vetores.\n', size(dict, 1));

    % --- 4. Empacotar e Retornar o Modelo ---
    model.dict = dict;
    model.alpha = akn;
    model.options = options; % Guarda os parâmetros para a predição
end
