%% RAG_BERT_contrastivo.m (CON PARADA TEMPRANA / EARLY STOPPING)
% MATLAB R2024a/R2024b. Solo codigo MATLAB y BERT multilingue oficial.
%
% INCLUYE:
%  - Entrenamiento en 2 epocas (1/2 y 2/2) con parada automatica si:
%      a) La perdida baja del umbral X (cfg.perdidaMinimaX).
%      b) La perdida no mejora tras N lotes seguidos (cfg.paciencia).
%  - Limpieza del PDF, exclusion del Indice y resolucion de las 50 preguntas.

clear; clc; rng(42);

%% CONFIGURACION
carpeta = fileparts(mfilename('fullpath'));
cfg.pdf = fullfile(carpeta,'documento.pdf');
cfg.k = 5;
cfg.palabrasPorChunk = 85;
cfg.solapePalabras = 25;
cfg.maxTokensBERT = 160;
cfg.loteInferencia = 8;

% --- CONFIGURACION DEL ENTRENAMIENTO Y PARADA AUTOMATICA ---
cfg.ajustarBERT = false;
cfg.epocas = 2;                 % Formato de 2 epocas (Epoca 1/2 y 2/2)
cfg.loteEntrenamiento = 6;
cfg.maxLotesPorEpoca = 25;      % Lotes maximos por epoca
cfg.tasaAprendizaje = 5e-6;     % Tasa de aprendizaje progresiva
cfg.temperatura = 0.10;
cfg.perdidaMinimaX = 0.01;      % FRENO 1 (Valor X): Si la perdida es menor que esto, PARA.
cfg.paciencia = 15;              % FRENO 2: Si en 15 lotes seguidos no mejora, PARA.
cfg.mejoraMinima = 0.005;       % Cuanto debe bajar la perdida para considerarse mejora

cfg.pesoSemanticoBERT = 0.35;
cfg.pesoTokensBERT = 0.65;
cfg.usarGPU = false;
if exist('canUseGPU','file') == 2
    cfg.usarGPU = canUseGPU;
end

%% BATERIA DE 50 PREGUNTAS ALINEADAS CON EL MANUAL DE MATLAB DEL PDF
preguntas = [
    "¿Cómo se define una función de usuario con la sentencia function y argumentos de entrada y salida?"
    "¿Cómo se define un vector fila y un vector columna?"
    "¿Para qué sirve el punto y coma ; al final de una sentencia en los ficheros de comandos?"
    "¿Cómo se calcula la matriz inversa con inv y la matriz traspuesta?"
    "¿Qué diferencia hay entre la multiplicación matricial * y el producto elemento a elemento .*?"
    "¿Cómo se genera un vector con valores igualmente espaciados usando linspace o logspace?"
    "¿Cómo se crea una matriz componiendo submatrices o accediendo a sus elementos con subíndices?"
    "¿Para qué sirven los comandos who, whos y clear en el espacio de trabajo?"
    "¿Cuál es la sintaxis de la sentencia for para ejecutar un bucle un número predeterminado de veces?"
    "¿Cuál es la estructura y sintaxis de la sentencia while ... end?"
    "¿Cómo se utilizan las sentencias condicionales if, elseif y else?"
    "¿Para qué sirve la bifurcación múltiple switch, case y otherwise?"
    "¿Cómo se dibuja una gráfica en 2D con la función plot y grid?"
    "¿Cómo se añaden título y etiquetas a los ejes con title, xlabel, ylabel y text?"
    "¿Para qué sirven los comandos hold on y hold off al superponer gráficas?"
    "¿Cómo se divide una ventana gráfica en múltiples subventanas con el comando subplot?"
    "¿Cómo se dibujan líneas y superficies en tres dimensiones con plot3, mesh y surf?"
    "¿Cómo se guardan y recuperan variables en disco con los comandos save y load?"
    "¿Qué es un fichero de comandos o script .m y para qué sirve el comando echo?"
    "¿Cómo se crean matrices de ceros con zeros, de unos con ones y la matriz unidad con eye?"
    "¿Cómo se resuelven sistemas de ecuaciones lineales Ax = b con el operador barra invertida \?"
    "¿Cómo se calculan los valores propios y vectores propios de una matriz cuadrada con eig?"
    "¿Cómo se convierten y manejan cadenas de caracteres con char y double?"
    "¿Qué es un vector de celdas o cell array y cómo se crean sus elementos con llaves?"
    "¿Cómo se definen y utilizan las estructuras struct y sus campos fields?"
    "¿Cómo se abren y cierran ficheros con fopen y fclose para lectura y escritura?"
    "¿Cómo se leen y escriben datos con formato mediante fscanf, sscanf, fprintf y sprintf?"
    "¿Cómo se leen variables desde teclado de forma interactiva con la función input?"
    "¿Qué son las funciones anónimas definidas con el símbolo arroba @ y un fhandle?"
    "¿Cómo se calculan las raíces de un polinomio con roots y su polinomio característico con poly?"
    "¿Cómo se ajusta un polinomio por mínimo error cuadrático medio con polyfit y se deriva con polyder?"
    "¿Cómo se calcula el valor interpolado de un conjunto de puntos con interp1?"
    "¿Cómo se calcula la integral definida de una función con quad?"
    "¿Qué funciones de MATLAB se utilizan para integrar sistemas de ecuaciones diferenciales rígidos y no rígidos, y cuál es el uso de ode45?"
    "¿Cómo operan las funciones sum, max y min sobre vectores, matrices e hipermatrices?"
    "¿Cómo se buscan los índices de elementos de vectores que cumplen una condición con find, any y all?"
    "¿Para qué sirven los operadores lógicos breves && y || combinados con operadores relacionales?"
    "¿Cómo se mide el tiempo necesario para ejecutar un cálculo con tic, toc, clock y cputime?"
    "¿Cómo se depura un fichero paso a paso con Step, Step In, Step Out y Breakpoints en el Debugger?"
    "¿Cómo funciona la construcción try, catch y end para gestionar errores en tiempo de ejecución?"
    "¿Qué son las variables locales y las variables persistentes declaradas con persistent?"
    "¿Cómo se cambia el formato numérico en pantalla con format long, format compact y format loose?"
    "¿Cómo se forma una matriz de números aleatorios entre 0 y 1 con distribución uniforme o normal?"
    "¿Para qué sirven las funciones eval y evalc al evaluar una cadena de caracteres como expresión?"
    "¿Cómo se calculan las raíces de ecuaciones no lineales con fzero y el mínimo con fminbnd y fminsearch?"
    "¿Cómo se crea una cuadrícula de puntos en el plano con la función meshgrid?"
    "¿Para qué sirven las funciones size, length y diag aplicadas a matrices y vectores?"
    "¿Cómo funciona el comando lookfor para buscar una palabra en la primera línea de comentario H1 y help?"
    "¿Cómo se calcula la descomposición en valores singulares svd, la pseudo-inversa pinv y el rango rank?"
    "¿Cómo se evalúa una función pasada como argumento mediante feval o un function handle?"
];

%% CARGA DEL MODELO BASE
% modelo = fullfile(carpeta,'bert_multilingue_contrastivo.mat');
% if ~isfile(modelo)
%     error('No existe el modelo. Ejecuta primero 01_preparar_BERT_multilingue_R2024.m');
% end
% s = load(modelo,'net','tokenizer');
% net = s.net;
% tokenizer = s.tokenizer;

modeloBase = fullfile(carpeta,'bert_multilingue_base.mat');
% modeloCLS = fullfile(carpeta,'bert_multilingue_contrastivo_CLS.mat');
modeloMEAN = fullfile(carpeta,'bert_multilingue_contrastivo_MEAN.mat');

% Si vamos a entrenar, partimos del BERT original.
% Si no vamos a entrenar, cargamos el BERT contrastivo con CLS.
if cfg.ajustarBERT
    modelo = modeloBase;
else
    % modelo = modeloCLS;
     modelo = modeloMEAN;
end

if ~isfile(modelo)
    error('No existe el modelo: %s',modelo);
end

s = load(modelo,'net','tokenizer');
net = s.net;
tokenizer = s.tokenizer;

%% 1. LEER Y LIMPIAR EL PDF
if ~isfile(cfg.pdf)
    [f,p] = uigetfile('*.pdf','Selecciona el manual de MATLAB en PDF');
    if isequal(f,0)
        error('No se ha seleccionado ningun PDF.');
    end
    cfg.pdf = fullfile(p,f);
end
textoBruto = extractFileText(cfg.pdf);
texto = limpiarTextoPDF(textoBruto);
fprintf('Documento: %s\n',cfg.pdf);
fprintf('Caracteres limpios extraidos: %d\n',strlength(texto));

%% 2. CHUNKS CON SOLAPAMIENTO (DESCARTANDO LA TABLA DE CONTENIDOS)
chunksTodos = generarChunks(texto,cfg.palabrasPorChunk,cfg.solapePalabras);
chunks = filtrarChunksIndice(chunksTodos);
N = numel(chunks);
fprintf('Chunks utiles generados (sin paginas de indice): %d\n',N);

%% 3. AJUSTE FINO CONTRASTIVO CON PARADA AUTOMATICA (EARLY STOPPING)
if cfg.ajustarBERT && N >= 2
    fprintf('Iniciando entrenamiento contrastivo de BERT (Max %d epocas | Objetivo X <= %.2f | Paciencia = %d)...\n', ...
        cfg.epocas, cfg.perdidaMinimaX, cfg.paciencia);
    promedio = [];
    promedio2 = [];
    iteracion = 0;
    numeroLotes = min(floor(N/cfg.loteEntrenamiento), cfg.maxLotesPorEpoca);
    
    mejorPerdida = inf;
    mejorNet = net;
    lotesSinMejora = 0;
    pararEntrenamiento = false;
    
    for epoca = 1:cfg.epocas
        orden = randperm(N);
        for j = 1:numeroLotes
            sel = orden((j-1)*cfg.loteEntrenamiento+1 : j*cfg.loteEntrenamiento);
            [ids,masc,seg] = codificarLote(tokenizer,chunks(sel),cfg.maxTokensBERT);
            idsDoble = cat(3,ids,ids);
            mascDoble = cat(3,masc,masc);
            segDoble = cat(3,seg,seg);
            entradas = entradasBERT(net,idsDoble,mascDoble,segDoble,cfg.usarGPU);
            
            [perdida,gradientes] = dlfeval(@gradienteContrastivo,net, ...
                entradas,mascDoble,cfg.loteEntrenamiento,cfg.temperatura);
            
            iteracion = iteracion + 1;
            [net,promedio,promedio2] = adamupdate(net,gradientes, ...
                promedio,promedio2,iteracion,cfg.tasaAprendizaje);
            
            valPerdida = double(gather(extractdata(perdida)));
            
            % Comprobar si la perdida ha mejorado respecto al mejor valor historico
            if valPerdida < (mejorPerdida - cfg.mejoraMinima)
                mejorPerdida = valPerdida;
                mejorNet = net;
                lotesSinMejora = 0;
                estadoMejora = "mejora OK";
            else
                lotesSinMejora = lotesSinMejora + 1;
                estadoMejora = sprintf("sin mejora %d/%d", lotesSinMejora, cfg.paciencia);
            end
            
            fprintf('Epoca %d/%d | Lote %d/%d | Perdida InfoNCE: %.5f (%s)\n', ...
                epoca, cfg.epocas, j, numeroLotes, valPerdida, estadoMejora);
            
            % CONDICION DE PARADA 1: La perdida ya es menor que X
            if valPerdida <= cfg.perdidaMinimaX
                fprintf('--> [PARADA POR OBJETIVO] La perdida (%.5f) es menor que X (%.2f). Entrenamiento completado.\n', ...
                    valPerdida, cfg.perdidaMinimaX);
                pararEntrenamiento = true;
                break;
            end
            
            % CONDICION DE PARADA 2: Lleva demasiados lotes sin mejorar
            if lotesSinMejora >= cfg.paciencia
                fprintf('--> [PARADA POR PACIENCIA] No hay mejora tras %d lotes seguidos. Se restaura el mejor modelo (perdida %.5f).\n', ...
                    cfg.paciencia, mejorPerdida);
                net = mejorNet;
                pararEntrenamiento = true;
                break;
            end
        end
        if pararEntrenamiento
            break;
        end
    end
    % rutaAjustada = fullfile(carpeta,'bert_multilingue_contrastivo.mat');
    % save(rutaAjustada,'net','tokenizer','cfg','-v7.3');
    % rutaAjustada = fullfile(carpeta,'bert_multilingue_contrastivo_CLS.mat');
    rutaAjustada = fullfile(carpeta,'bert_multilingue_contrastivo_MEAN.mat');
    net = mejorNet;
    
    rutaAjustada = fullfile(carpeta,'bert_multilingue_contrastivo_MEAN.mat');
    save(rutaAjustada,'net','tokenizer','cfg','-v7.3');

    fprintf('Restaurando el mejor modelo. Perdida: %.5f\n', ...
        mejorPerdida);
end

%% 4. INDEXAR CHUNKS (EMBEDDINGS NEURONALES + SECUENCIAS DE TOKENS BERT)
fprintf('Calculando embeddings BERT de los %d chunks...\n', N);
embeddingsBrutos = generarEmbeddings(net,tokenizer,chunks, ...
    cfg.maxTokensBERT,cfg.loteInferencia,cfg.usarGPU);

mediaGlobal = mean(embeddingsBrutos, 2);
embeddings = embeddingsBrutos - mediaGlobal;
embeddings = embeddings ./ max(vecnorm(embeddings,2,1), eps);

fprintf('Indexando secuencias de tokens de BERT...\n');
[secuenciasTokensChunk, idfTokens] = indexarSecuenciasBERT(tokenizer, chunks);

%% 5. RESOLVER LAS 50 PREGUNTAS
numPreguntas = numel(preguntas);
fprintf('\n======================================================\n');
fprintf('RESOLVIENDO LAS %d PREGUNTAS DEL MANUAL DE MATLAB\n', numPreguntas);
fprintf('======================================================\n');

preguntasLimpias = limpiarPreguntasParaBusqueda(preguntas);
vecPreguntasBruto = generarEmbeddings(net,tokenizer,preguntasLimpias, ...
    cfg.maxTokensBERT,cfg.loteInferencia,cfg.usarGPU);
vecPreguntas = vecPreguntasBruto - mediaGlobal;
vecPreguntas = vecPreguntas ./ max(vecnorm(vecPreguntas,2,1), eps);

archivoSalida = fullfile(carpeta,'respuestas_50_preguntas_MATLAB_MEAN.txt');
fid = fopen(archivoSalida,'w','n','UTF-8');

k = min(cfg.k, N);
for p = 1:numPreguntas
    consultaOriginal = preguntas(p);
    consultaLimpia = preguntasLimpias(p);
    
    % 1) Similitud coseno neuronal con BERT
    simSemantica = embeddings.' * vecPreguntas(:,p);
    
    % 2) Coincidencia de secuencias de tokens BERT ponderada por IDF^2
    simTokens = puntuarTokensBERT(tokenizer, consultaLimpia, ...
        chunks, secuenciasTokensChunk, idfTokens);
    
    % Puntuacion combinada 100% BERT
    puntuacion = cfg.pesoSemanticoBERT * simSemantica + cfg.pesoTokensBERT * simTokens;
    [valores, orden] = sort(puntuacion, 'descend');

    if ismember(p,[1 24 46])

    fprintf('\n========== ANALISIS DE RECUPERACION ==========\n');

    for j = 1:min(5,N)

        idx = orden(j);

        fprintf('\nTOP %d | CHUNK %d | SCORE %.4f\n', ...
            j,idx,valores(j));

        fprintf('%s\n',chunks(idx));

    end

    fprintf('\n==============================================\n');

end
    
    chunkGanador = chunks(orden(1));
    
    fprintf('\n------------------------------------------------------\n');
    fprintf('PREGUNTA %d: %s\n', p, consultaOriginal);
    fprintf('RESPUESTA EXTRAIDA:\n  %s\n', chunkGanador);
    fprintf('  [Chunk #%d | Puntuacion BERT: %.4f]\n', orden(1), valores(1));
    
    fprintf(fid, '======================================================\n');
    fprintf(fid, 'PREGUNTA %d: %s\n', p, consultaOriginal);
    fprintf(fid, 'RESPUESTA (Chunk #%d | Puntuacion BERT: %.4f):\n%s\n\n', ...
        orden(1), valores(1), chunkGanador);
    for i = 1:k
        idx = orden(i);
        fprintf(fid, '  --- Top %d (Chunk %d | Score %.4f) ---\n  %s\n\n', ...
            i, idx, valores(i), chunks(idx));
    end
end
fclose(fid);

%% 6. GUARDAR INDICE
datosModelo = dir( fullfile(carpeta,'bert_multilingue_contrastivo_MEAN.mat') );
fechaModeloIndexado = datosModelo.datenum;
rutaIndice = fullfile(carpeta,'indice_RAG_BERT_MEAN.mat');
save(rutaIndice,'chunks','embeddings','mediaGlobal','cfg', ...
    'secuenciasTokensChunk','idfTokens','fechaModeloIndexado','-v7.3');

%% ========================================================================
%% FUNCIONES LOCALES (SOLO MATLAB + BERT)
%% ========================================================================

function texto = limpiarTextoPDF(textoBruto)
    texto = string(textoBruto);
    texto = regexprep(texto, '\s+', ' ');
    texto = regexprep(texto, 'Aprenda Matlab\s*7\.0\s*como si estuviera en Primero\s*página\s*[0-9ivx]+', ' ', 'ignorecase');
    texto = regexprep(texto, 'Capítulo\s*\d+:[^.]{0,55}?página\s*\d+', ' ', 'ignorecase');
    texto = regexprep(texto, 'Índice\s*página\s*[0-9ivx]+', ' ', 'ignorecase');
    texto = regexprep(texto, 'Prólogo\s*página\s*\d+', ' ', 'ignorecase');
    texto = regexprep(texto, '([a-záéíóúñA-ZÁÉÍÓÚÑ])-\s+([a-záéíóúñ])', '$1$2');
    texto = regexprep(texto, '\s+', ' ');
    texto = strtrim(texto);
end

function chunksUtiles = filtrarChunksIndice(chunks)
    mantener = true(numel(chunks), 1);
    for i = 1:numel(chunks)
        c = chunks(i);
        numSubapartados = numel(regexp(c, '\<\d+\.\d+\.\d+\.'));
        numLineasIndice = numel(regexp(c, '\<\d+\.\d+(\.\d+)?\.\s+[^\d]{3,60}\s+\d+\>'));
        if numSubapartados >= 3 || numLineasIndice >= 2
            mantener(i) = false;
        end
    end
    chunksUtiles = chunks(mantener);
end

function preguntasLimpias = limpiarPreguntasParaBusqueda(preguntas)
    relleno = [
        "¿", "?", "¡", "!", ",", ...
        "cómo", "como", "qué", "que", "cuál", "cual", "para", "sirve", "sirven", ...
        "hay", "entre", "son", "es", "se", "en", "el", "la", "los", "las", ...
        "un", "una", "unos", "unas", "de", "del", "al", "con", "por", "su", "sus", ...
        "o", "y", "a", "matlab"
    ];
    preguntasLimpias = preguntas;
    for i = 1:numel(preguntas)
        q = lower(preguntas(i));
        q = replace(q, ["¿","?","¡","!",","], " ");
        palabras = split(strtrim(q));
        palabras = palabras(strlength(palabras) > 0);
        mantener = ~ismember(palabras, relleno);
        if any(mantener)
            preguntasLimpias(i) = join(palabras(mantener), " ");
        end
    end
end

function chunks = generarChunks(texto,nPalabras,solape)
    palabras = split(strtrim(texto));
    palabras = palabras(strlength(palabras) > 0);
    total = numel(palabras);
    paso = nPalabras - solape;
    chunks = strings(0,1);
    for comienzo = 1:paso:total
        final = min(total, comienzo + nPalabras - 1);
        chunks(end+1,1) = join(palabras(comienzo:final),' '); %#ok<AGROW>
        if final == total
            break;
        end
    end
end

function [ids,masc,seg] = codificarLote(tokenizer,textos,maxTokens)
    textos = string(textos(:));
    B = numel(textos);
    ids = repmat(single(tokenizer.PaddingCode),[1,maxTokens,B]);
    masc = zeros(1,maxTokens,B,'single');
    seg = ones(1,maxTokens,B,'single');
    [codigos,segmentos] = encode(tokenizer,textos);
    for i = 1:B
        a = codigos{i};
        b = segmentos{i};
        if numel(a) > maxTokens
            a = [a(1:maxTokens-1), tokenizer.SeparatorCode];
            b = [b(1:maxTokens-1), b(end)];
        end
        L = numel(a);
        ids(1,1:L,i) = single(a);
        masc(1,1:L,i) = 1;
        seg(1,1:L,i) = single(b);
    end
end

function entradas = entradasBERT(net,ids,masc,seg,usarGPU)
    Xids = dlarray(ids,'CTB');
    Xmask = dlarray(masc,'CTB');
    Xseg = dlarray(seg,'CTB');
    if usarGPU
        Xids = gpuArray(Xids);
        Xmask = gpuArray(Xmask);
        Xseg = gpuArray(Xseg);
    end
    nombres = lower(string(net.InputNames));
    entradas = cell(1,numel(nombres));
    for i = 1:numel(nombres)
        nom = nombres(i);
        if contains(nom,'mask')
            entradas{i} = Xmask;
        elseif contains(nom,'seg') || contains(nom,'token_type')
            entradas{i} = Xseg;
        elseif contains(nom,'id') || contains(nom,'input')
            entradas{i} = Xids;
        else
            error('Entrada BERT no reconocida: %s',nom);
        end
    end
end

% MEAN POOLING

function M = generarEmbeddings(net,tokenizer,textos,maxTokens,lote,usarGPU)

    textos = string(textos(:));
    n = numel(textos);
    M = [];

    for primero = 1:lote:n

        ult = min(n, primero + lote - 1);

        [ids,masc,seg] = codificarLote(tokenizer,textos(primero:ult),maxTokens);

        entradas = entradasBERT(net,ids,masc,seg,usarGPU);

        Y = predict(net,entradas{:});

        % MEAN POOLING: promediar las representaciones de los tokens
        dimT = finddim(Y,'T');

        Xmask = dlarray(masc,'CTB');

        if usarGPU
            Xmask = gpuArray(Xmask);
        end

        Yraw = stripdims(Y);
        Mraw = stripdims(Xmask);

        sumaTokens = sum(Yraw .* Mraw, dimT);
        numTokens = max(sum(Mraw, dimT), 1e-8);

        media = sumaTokens ./ numTokens;

        D = size(Yraw,1);
        E = reshape(media,D,[]);

        % Pasar los embeddings a CPU y normalizarlos
        E = gather(extractdata(E));
        E = double(E);

        E = E ./ max(vecnorm(E,2,1),eps);

        if isempty(M)
            M = zeros(D,n,'double');
        end

        M(:,primero:ult) = E;

    end
end


% CLS
% function M = generarEmbeddings(net,tokenizer,textos,maxTokens,lote,usarGPU)
% 
%     textos = string(textos(:));
%     n = numel(textos);
%     M = [];
% 
%     for primero = 1:lote:n
% 
%         ult = min(n, primero + lote - 1);
% 
%         [ids,masc,seg] = codificarLote(tokenizer,textos(primero:ult),maxTokens);
% 
%         entradas = entradasBERT(net,ids,masc,seg,usarGPU);
% 
%         Y = predict(net,entradas{:});
% 
%         % Obtener las representaciones de BERT
%         dimT = finddim(Y,'T');
%         Yraw = stripdims(Y);
% 
%         % Seleccionar exclusivamente el primer token [CLS]
%         indices = repmat({':'},1,ndims(Yraw));
%         indices{dimT} = 1;
% 
%         CLS = Yraw(indices{:});
% 
%         % Obtener un vector por cada texto del lote
%         D = size(Yraw,1);
%         E = reshape(CLS,D,[]);
% 
%         % Pasar los embeddings a CPU y normalizarlos
%         E = gather(extractdata(E));
%         E = double(E);
% 
%         E = E ./ max(vecnorm(E,2,1),eps);
% 
%         if isempty(M)
%             M = zeros(D,n,'double');
%         end
% 
%         M(:,primero:ult) = E;
% 
%     end
% end

%CLS

% function [perdida,gradientes] = gradienteContrastivo(net,entradas,mascDoble,B,tau)
% 
%     % Forward de BERT durante el entrenamiento
%     Y = forward(net,entradas{:});
% 
%     % Eliminar etiquetas de dimensiones
%     Yraw = stripdims(Y);
% 
%     % Seleccionar exclusivamente el primer token [CLS]
%     dimT = finddim(Y,'T');
% 
%     indices = repmat({':'},1,ndims(Yraw));
%     indices{dimT} = 1;
% 
%     Z = Yraw(indices{:});
% 
%     % Obtener una representación por cada texto del lote
%     C = size(Yraw,1);
%     Z = reshape(Z,C,2*B);
% 
%     % Normalizar las representaciones
%     Z = Z ./ sqrt(sum(Z.^2,1) + 1e-8);
% 
%     % Calcular similitudes entre representaciones
%     logits = (Z.' * Z) / tau;
% 
%     n = 2*B;
%     mascaraDiagonal = eye(n,'single');
%     positivos = zeros(n,n,'single');
% 
%     % Cada fragmento tiene como positivo su segunda representación
%     for i = 1:B
%         positivos(i,i+B) = 1;
%         positivos(i+B,i) = 1;
%     end
% 
%     if isa(extractdata(logits),'gpuArray')
%         mascaraDiagonal = gpuArray(mascaraDiagonal);
%         positivos = gpuArray(positivos);
%     end
% 
%     % Evitar que cada representación se compare consigo misma
%     logits = logits - 1e4 * mascaraDiagonal;
% 
%     maxFilas = max(logits,[],2);
%     logNormalizador = maxFilas + log(sum(exp(logits - maxFilas),2));
% 
%     logitPositivo = sum(logits .* positivos,2);
% 
%     % Pérdida contrastiva InfoNCE
%     perdida = mean(logNormalizador - logitPositivo,'all');
% 
%     % Calcular gradientes de la red
%     gradientes = dlgradient(perdida,net.Learnables);
% 
% end

% MEAN POOLING


function [perdida,gradientes] = gradienteContrastivo(net,entradas,mascDoble,B,tau)

    Y = forward(net,entradas{:});

    dimT = finddim(Y,'T');

    Xmask = dlarray(mascDoble,'CTB');

    if isa(extractdata(Y),'gpuArray')
        Xmask = gpuArray(Xmask);
    end

    Yraw = stripdims(Y);
    Mraw = stripdims(Xmask);

    % MEAN POOLING: promedio de los tokens validos
    Z = sum(Yraw .* Mraw,dimT) ./ max(sum(Mraw,dimT),1e-8);

    C = size(Yraw,1);
    Z = reshape(Z,C,2*B);

    % Normalizar las representaciones
    Z = Z ./ sqrt(sum(Z.^2,1) + 1e-8);

    logits = (Z.' * Z) / tau;

    n = 2*B;
    mascaraDiagonal = eye(n,'single');
    positivos = zeros(n,n,'single');

    for i = 1:B
        positivos(i,i+B) = 1;
        positivos(i+B,i) = 1;
    end

    if isa(extractdata(logits),'gpuArray')
        mascaraDiagonal = gpuArray(mascaraDiagonal);
        positivos = gpuArray(positivos);
    end

    logits = logits - 1e4 * mascaraDiagonal;

    maxFilas = max(logits,[],2);
    logNormalizador = maxFilas + log(sum(exp(logits - maxFilas),2));

    logitPositivo = sum(logits .* positivos,2);

    perdida = mean(logNormalizador - logitPositivo,'all');

    gradientes = dlgradient(perdida,net.Learnables);

end


function [secuenciasChunk, idfTokens] = indexarSecuenciasBERT(tokenizer, chunks)
    N = numel(chunks);
    codigos = encode(tokenizer, lower(chunks));
    secuenciasChunk = cell(N,1);
    especiales = double([tokenizer.PaddingCode, tokenizer.StartCode, tokenizer.SeparatorCode]);
    
    maxId = 150000;
    df = zeros(maxId, 1);
    for i = 1:N
        seq = double(codigos{i}(:).');
        seq = seq(~ismember(seq, especiales));
        secuenciasChunk{i} = seq;
        u = unique(seq);
        u = u(u > 0 & u <= maxId);
        df(u) = df(u) + 1;
    end
    idfBase = log((N + 1) ./ (df + 1));
    idfTokens = idfBase .^ 2;
end

function scores = puntuarTokensBERT(tokenizer, consultaLimpia, chunks, secuenciasChunk, idfTokens)
    N = numel(secuenciasChunk);
    scores = zeros(N,1);
    especiales = double([tokenizer.PaddingCode, tokenizer.StartCode, tokenizer.SeparatorCode]);
    
    terminos = unique(split(strtrim(lower(consultaLimpia))), 'stable');
    terminos = terminos(strlength(terminos) > 0);
    if isempty(terminos)
        return;
    end
    
    numTerm = numel(terminos);
    seqTerminos = cell(numTerm, 1);
    pesoTermino = zeros(numTerm, 1);
    codTerm = encode(tokenizer, terminos);
    
    for t = 1:numTerm
        seqT = double(codTerm{t}(:).');
        seqT = seqT(~ismember(seqT, especiales));
        seqTerminos{t} = seqT;
        idsValidos = seqT(seqT > 0 & seqT <= numel(idfTokens));
        if ~isempty(idsValidos)
            pesoTermino(t) = max(idfTokens(idsValidos));
        end
    end
    
    pesoTotal = sum(pesoTermino) + eps;
    chunksLower = lower(chunks);
    
    for i = 1:N
        seqC = secuenciasChunk{i};
        puntosChunk = 0;
        terminosAcertados = 0;
        for t = 1:numTerm
            seqT = seqTerminos{t};
            if isempty(seqT)
                continue;
            end
            acierto = contieneSecuencia(seqC, seqT); %#ok<STREMP>
            if acierto && strlength(terminos(t)) >= 2 && isletter(char(extractBetween(terminos(t),1,1)))
                patron = "\<" + regexptranslate('escape', terminos(t)) + "\>";
                acierto = ~isempty(regexp(chunksLower(i), patron, 'once'));
            end
            if acierto
                puntosChunk = puntosChunk + pesoTermino(t);
                terminosAcertados = terminosAcertados + 1;
            end
        end
        proporcion = puntosChunk / pesoTotal;
        bonus = 1 + 0.12 * max(0, terminosAcertados - 1);
        scores(i) = min(1.0, proporcion * bonus);
    end
end


function encontrado = contieneSecuencia(secuencia, patron)

    secuencia = double(secuencia(:).');
    patron = double(patron(:).');

    encontrado = false;

    n = numel(patron);
    m = numel(secuencia);

    if n == 0 || n > m
        return;
    end

    for i = 1:(m-n+1)

        if all(secuencia(i:i+n-1) == patron)

            encontrado = true;
            return;

        end

    end

end
