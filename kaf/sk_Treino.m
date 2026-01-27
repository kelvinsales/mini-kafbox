function model = sk_Treino(X, Y, options)
% Implementacao S-KRLS baseado no Artigo de HAIJIN FAN 2013/KAF box
%
% Objetivo: Aplicacao do S-KRLS em predicao
%
% Autor: Allan Kelvin M Sales
% Data: 24/11/2025

% Treina um modelo SKRLS (SPARSE - Kernel Recursive Least Squares).
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
    [delta, beta] = options.hiperpar{:};
    [k_tipo, k_gamma] = options.kernel{:};

    [Nx, ~] = size(X);

    % --- 2. Inicializar o Modelo com a primeira amostra ---
    dict = X(1,:);
    aks   = Y(1); % Nota: a inicialização pode variar, esta é uma comum.
    P    = 1;
    sk_inst = 1;
    hist_Dic{1} = dict;
    hist_A{1} = aks;

    fprintf('Iniciando o treinamento KRLS para %d amostras...\n', Nx);
    tic;

    % --- 3. Loop de Treinamento Online (a partir da segunda amostra) ---
    for t = 2:Nx
        x_current = X(t, :);
        y_current = Y(t);

        % Vetor de kernels entre a amostra atual e o dicionário
        kel = kernel_K(dict, x_current, k_tipo, k_gamma);

        % Valores predito e observado
        dj = y_current;
        y_p = kel'*aks;


        % Erro a priori
        err_priori = dj-y_p;

        % Ganho de Kalman
        g = (P*kel)/(beta+kel'*P*kel);

        % Atualiza coeficiente atual
        aks = aks + g*err_priori;

        % Atualiza matriz de correlacao
        P = 1/beta*(P - (P*kel*kel'*P)/(beta+kel'*P*kel));

        % Valores observado a posteriori
        y_p = kel'*aks;

        % Erro a posteriori
        err_post = dj-y_p;

            % Variavel auxiliar mi
        mi = 1 - kel'*P*kel;

        % Proteção numérica para mi
        if mi <= 1e-12
            mi = 1e-12;
        endif

        % Computa o deltaL
        deltaL(t) = err_post.^2/(2*mi);

        if deltaL(t)>delta

          % Aumenta a ordem do dicionario
          dict = [dict; x_current];
          sk_inst = [sk_inst;t];

          % Aumenta a ordem de aks
          aks = [aks-(P*kel)*err_post/mi; err_post/mi];

          % Variavel auxiliar at
          at1 = (P*kel)/mi;
          at2 = (kel'*P)/mi;

        % Atualiza matriz de Projecao
          P = [P + (P*kel*kel'*P)/mi, -at1;-at2, 1/mi];
        endif
        hist_Dic{t} = dict;
        hist_A{t} = aks;
    endfor

    training_time = toc;
    fprintf('Treinamento concluído em %.3f segundos.\n', training_time);
    fprintf('Tamanho final do dicionário: %d vetores.\n', size(dict, 1));

    % --- 4. Empacotar e Retornar o Modelo ---
    model.dict = dict;
    model.alpha = aks;
    model.P = P;       % Salva o estado interno para possível continuação
    model.options = options;
    model.hist_Dic = hist_Dic;
    model.hist_A = hist_A;
    model.inst = sk_inst;

end
