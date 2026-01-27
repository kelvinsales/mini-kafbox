
function model = k_T_Treino(X, Y, options)
  % Implementacao KRLS - Tracker baseado no Artigo de VAERENBERGH 2012/KAF box
  %
  % Objetivo: Aplicacao da KTLS-T para predicao
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
  %       .k_gamma (k_gamma)
  %
  % Output:
  %   model: estrutura contendo o modelo treinado:
  %       .lambda:  Limiar do criterio de suficiencia
  %       .sn2:     Regularizacao
  %       .M:       Tamanho do dicionario
  %       .jitter:  Latência de Ruido para identificar contorno erro
  %       .dict:    o dicionário de instâncias.
  %       .alpha:   os coeficientes do modelo (ak).
  %       .Q:    a inversa da matriz de kernel do dicionário final.
  %       .P:       a matriz de projeção final.
  %       .options: os hiperparâmetros usados no treino.

  % --- 1. Extrair Hiperparâmetros e Tamanho da Série ---
  [lambda,sn2, jitter, M] = options.hiperpar{:};
  [k_tipo, k_gamma, bias] = options.kernel{:};

  [Nx, ~] = size(X);

  % --- 2. Inicializar o Modelo com valores iniciais ---
  dict = [];            % Dicionario
  Q = [];               % Matriz inversa de kernel_K
  Sigma = [];           % posterior covariance
  mu = [];              % posterior mean
  nums02ML = 0;
  dens02ML = 0;
  s02 = 0;              % signal power, adaptively estimated
  prune = false;        % flag
  reduced = false;      % flag

  % --- 3. Etapa de Treinamento Online ---
  fprintf('Iniciando o treinamento KRLS para %d amostras...\n', Nx);
  tic;

  for t=1:Nx

    m = size(Sigma,1);

    if m<1 % initialize
      kel = kernel_K(X(t,:),X(t,:),k_tipo,k_gamma,bias);
      kel = kel + jitter;
      Q = 1/kel;
      mu = Y(t,:)*kel/(kel+sn2);
      Sigma = kel - kel.^2/(kel+sn2);
      dict = X(t,:); % dictionary bases
      nums02ML = Y(t,:)^2/(kel+sn2);
      dens02ML = 1;
      s02 = nums02ML / dens02ML;
      hist_Dic{1} = 1;
      hist_A{1} = Q*mu;
    else
      % forget
      K = kernel_K(dict,dict,k_tipo,k_gamma,bias) + jitter*eye(m);
      Sigma = lambda*Sigma + (1-lambda)*K; % forget
      mu = sqrt(lambda)*mu; % forget

      % predict
      kel = kernel_K([dict; X(t,:)],X(t,:),k_tipo,k_gamma,bias);
      kt = kel(1:end-1);
      ktt = kel(end) + jitter;
      q = Q*kt;
      y_mean = q'*mu; % predictive mean
      gamma2 = ktt - kt'*q; gamma2(gamma2<0)=0; % projection uncertainty
      h = Sigma*q;
      sf2 = gamma2 + q'*h; sf2(sf2<0)=0; % noiseless prediction variance
      sy2 = sn2 + sf2;
      % y_var = s02*sy2; % predictive variance

      % include a new sample and add a basis
      Qold = Q; % old inverse kernel_K matrix
      p = [q; -1];
      Q = [Q zeros(m,1);zeros(1,m) 0] + 1/gamma2*(p*p');

      p = [h; sf2];
      mu = [mu; y_mean] + (Y(t,:) - y_mean)/sy2*p; % posterior mean
      Sigma = [Sigma h; h' sf2] - 1/sy2*(p*p'); % posterior covariance
      m = m + 1;
      dict = [dict; X(t,:)];

      % estimate s02 via ML
      nums02ML = nums02ML + lambda*(Y(t,:) - y_mean)^2/sy2;
      dens02ML = dens02ML + lambda;
      s02 = nums02ML/dens02ML;

      prune = false;
      % delete a basis if necessary
      if (m>M  || gamma2<jitter)
       if gamma2<jitter, % to avoid roundoff error
          if gamma2<jitter/10
             warning('Numerical roundoff error too high, you should increase jitter noise') %#ok<WNTAG>
          end
          criterion = [ones(1,m-1) 0];
       else % MSE pruning criterion
          errors = (Q*mu)./diag(Q);
          criterion = abs(errors);
       end
       [~, r] = min(criterion); % remove element r, which incurs in the minimum error
       smaller = 1:m; smaller(r) = [];

       if r == m, % if we must remove the element we just added, perform reduced update instead
        Q = Qold;
        reduced = true;
       else
        Qs = Q(smaller, r);
        qs = Q(r,r); Q = Q(smaller, smaller);
        Q = Q - (Qs*Qs')/qs;
        reduced = false;
       end
       mu = mu(smaller);
       Sigma = Sigma(smaller, smaller);
       dict = dict(smaller,:);
       prune = true;
      end
    end
    hist_Dic{t} = dict;
    hist_A{t} = Q*mu;
  endfor

  training_time = toc;
  fprintf('Treinamento concluído em %.2f segundos.\n', training_time);
  fprintf('Tamanho final do dicionário: %d vetores.\n', size(dict, 1));

  % --- 4. Empacotar e Retornar o Modelo ---
  model.dict = dict;
  model.alpha = Q*mu;
  model.Q = Q;        % Salva o estado interno para possível continuação
  model.mu = mu;      % Salva o estado interno para possível continuação
  model.options = options;
  model.hist_Dic = hist_Dic;
  model.hist_A = hist_A;
end
