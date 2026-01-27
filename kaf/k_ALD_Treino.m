function model = k_ALD_Treino(X, Y, options)
% Implementacao KRLS - ALD baseado no Artigo de ENGEL 2004/KAF box
%
% Objetivo: Aplicacao do KRLS - ALD para predicao
%
% Autor: Allan Kelvin M Sales
% Data: 24/11/2025

% Treina um modelo KRLS (Kernel Recursive Least Squares).
% Input:
%   X: n x d matriz de dados de treinamento.
%   Y: n x 1 vetor de rótulos de treinamento.
%   options: estrutura com os hiperparâmetros:
%       .limiar (nu)
%       .k_tipo ('gauss', etc.)
%       .k_gamma (gamma)
%
% Output:
%   model: estrutura contendo o modelo treinado:
%       .dict: o dicionário de vetores de suporte.
%       .alpha: os coeficientes do modelo (ak).
%       .Kinv: a inversa da matriz de kernel do dicionário final.
%       .P: a matriz de projeção final.
%       .options: os hiperparâmetros usados no treino.

    % --- 1. Extrair Hiperparâmetros ---
    % --- 1. Extrair Hiperparâmetros ---
    [nu] = options.hiperpar{:};
    [k_tipo, k_gamma] = options.kernel{:};

    [Nx, ~] = size(X);
    if Nx == 0
        error('Dados de treinamento não podem estar vazios.');
    end

    % --- 2. Inicializar o Modelo com a primeira amostra ---
    dict = X(1,:);
    k11  = kernel_K(dict, dict, k_tipo, k_gamma);
    Kinv = 1 / k11;
    ak   = Y(1) / k11; % Nota: a inicialização pode variar, esta é uma comum.
    P    = 1;
    hist_Dic{1} = 1;
    hist_A{1} = ak;

    fprintf('Iniciando o treinamento KRLS para %d amostras...\n', Nx);
    tic;

    % --- 3. Loop de Treinamento Online (a partir da segunda amostra) ---
    for t = 2:Nx
        x_current = X(t, :);
        y_current = Y(t);

        % Vetor de kernels entre a amostra atual e o dicionário
        kt = kernel_K(dict, x_current, k_tipo, k_gamma);

        % Auto-kernel da amostra atual
        ktt = kernel_K(x_current, x_current, k_tipo, k_gamma);

        % Projeção da nova amostra no espaço do dicionário
        at = Kinv * kt;

        % Erro de projeção (a medida de "novidade")
        delta = ktt - kt' * at;

        % Erro de predição do modelo atual
        pred_error = y_current - kt' * ak;

        % Critério de novidade para crescer o dicionário
        if delta > nu
            % --- Aumentar a ordem do modelo ---
            % Adicionar amostra ao dicionário
            dict = [dict; x_current];

            % Atualizar a inversa da matriz de kernel eficientemente
            Kinv = (1/delta) * [delta*Kinv + at*at', -at; -at', 1];

            % Atualizar matriz de projeção
            Ze = zeros(size(P,1), 1);
            P = [P, Ze; Ze', 1];

            % Calcular e atualizar os coeficientes
            ode = (1/delta) * pred_error;
            ak = [ak - at * ode; ode];
        else
            % --- Manter a ordem, apenas atualizar os coeficientes ---
            q = P * at / (1 + at' * P * at);
            P = P - q * (at' * P); % Atualizar matriz de projeção
            ak = ak + Kinv * q * pred_error; % Atualização de baixo custo
        end
        hist_Dic{t} = dict;
        hist_A{t} = ak;
    endfor

    training_time = toc;
    fprintf('Treinamento concluído em %.3f segundos.\n', training_time);
    fprintf('Tamanho final do dicionário: %d vetores.\n', size(dict, 1));

    % --- 4. Empacotar e Retornar o Modelo ---
    model.dict = dict;
    model.alpha = ak;
    model.Kinv = Kinv; % Salva o estado interno para possível continuação
    model.P = P;       % Salva o estado interno para possível continuação
    model.options = options;
    model.hist_Dic = hist_Dic;
    model.hist_A = hist_A;

end
