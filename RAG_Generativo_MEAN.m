%% RAG_Generativo_MEAN.m
% RAG generativo: recuperacion BERT Mean Pooling + generacion local con Ollama.
% Compatible con MATLAB R2024a/R2024b.
%
% Requisitos en la misma carpeta:
%   - bert_multilingue_contrastivo_MEAN.mat
%   - indice_RAG_BERT_MEAN.mat
%   - Ollama en ejecucion con llama3.2:latest instalado

clear; clc;

%% 1. CONFIGURACION
carpeta = fileparts(mfilename('fullpath'));
modeloBERT = fullfile(carpeta,'bert_multilingue_contrastivo_MEAN.mat');
rutaIndice = fullfile(carpeta,'indice_RAG_BERT_MEAN.mat');
modeloOllama = 'llama3.2:latest';
urlOllama = 'http://localhost:11434/api/chat';
k = 5;                         % Numero de chunks que se envian al LLM
loteInferencia = 8;
tiempoEsperaOllama = 180;       % Segundos

if ~isfile(modeloBERT)
    error('No se encuentra el modelo BERT: %s',modeloBERT);
end
if ~isfile(rutaIndice)
    error(['No se encuentra el indice: %s\n' ...
        'Ejecuta primero 02_RAG_BERT_contrastivo_R2024.m con el modelo MEAN.'],rutaIndice);
end

%% 2. CARGAR MODELO BERT E INDICE
fprintf('Cargando modelo BERT Mean Pooling...\n');
s = load(modeloBERT,'net','tokenizer');
net = s.net;
tokenizer = s.tokenizer;

I = load(rutaIndice);
camposNecesarios = {'chunks','cfg'};
for i = 1:numel(camposNecesarios)
    if ~isfield(I,camposNecesarios{i})
        error('El archivo de indice no contiene el campo requerido: %s',camposNecesarios{i});
    end
end
chunks = I.chunks;
cfg = I.cfg;

if isfield(cfg,'maxTokensBERT')
    maxTokensBERT = cfg.maxTokensBERT;
else
    maxTokensBERT = 160;
end
if isfield(cfg,'usarGPU')
    usarGPU = cfg.usarGPU;
else
    usarGPU = false;
end
if isfield(cfg,'pesoSemanticoBERT')
    pesoSemanticoBERT = cfg.pesoSemanticoBERT;
else
    pesoSemanticoBERT = 0.35;
end
if isfield(cfg,'pesoTokensBERT')
    pesoTokensBERT = cfg.pesoTokensBERT;
else
    pesoTokensBERT = 0.65;
end

datosModelo = dir(modeloBERT);
fechaModelo = datosModelo.datenum;
indiceActualizado = isfield(I,'fechaModeloIndexado') && ...
    isequal(I.fechaModeloIndexado,fechaModelo) && ...
    isfield(I,'embeddings') && isfield(I,'mediaGlobal') && ...
    isfield(I,'secuenciasTokensChunk') && isfield(I,'idfTokens');

if ~indiceActualizado
    fprintf('\nActualizando indice para que coincida con el modelo BERT cargado...\n');
    fprintf('Calculando embeddings de %d chunks. Esto solo sera necesario cuando cambie el modelo.\n',numel(chunks));
    embeddingsBrutos = generarEmbeddings(net,tokenizer,chunks, ...
        maxTokensBERT,loteInferencia,usarGPU);
    mediaGlobal = mean(embeddingsBrutos,2);
    embeddings = embeddingsBrutos - mediaGlobal;
    embeddings = embeddings ./ max(vecnorm(embeddings,2,1),eps);
    fprintf('Indexando secuencias de tokens...\n');
    [secuenciasTokensChunk,idfTokens] = indexarSecuenciasBERT(tokenizer,chunks);
    fechaModeloIndexado = fechaModelo;
    save(rutaIndice,'chunks','embeddings','mediaGlobal','cfg', ...
        'secuenciasTokensChunk','idfTokens','fechaModeloIndexado','-v7.3');
    fprintf('Indice actualizado y guardado en: %s\n',rutaIndice);
else
    embeddings = I.embeddings;
    mediaGlobal = I.mediaGlobal;
    secuenciasTokensChunk = I.secuenciasTokensChunk;
    idfTokens = I.idfTokens;
    fprintf('Indice cargado: %d chunks.\n',numel(chunks));
end

N = numel(chunks);
if N == 0
    error('El indice no contiene chunks.');
end
k = min(k,N);

%% 3. COMPROBAR CONEXION CON OLLAMA
fprintf('\nComprobando Ollama (%s)...\n',modeloOllama);
try
    opciones = weboptions('MediaType','application/json','Timeout',15);
    modelos = webread('http://localhost:11434/api/tags',opciones);
catch ME
    error(['No se puede conectar con Ollama. Comprueba que Ollama esta abierto ' ...
        'y que escucha en http://localhost:11434.\nDetalle: %s'],ME.message);
end
modeloDisponible = false;
if isfield(modelos,'models')
    for i = 1:numel(modelos.models)
        if isfield(modelos.models(i),'name') && strcmp(modelos.models(i).name,modeloOllama)
            modeloDisponible = true;
            break;
        end
    end
end
if ~modeloDisponible
    error('El modelo %s no aparece instalado en Ollama. Ejecuta: ollama pull %s', ...
        modeloOllama,modeloOllama);
end
fprintf('Ollama conectado correctamente.\n');

%% 4. BUCLE DE CONSULTAS
fprintf('\n======================================================\n');
fprintf('RAG GENERATIVO - BERT MEAN POOLING + OLLAMA\n');
fprintf('Escribe una pregunta. Para terminar, escribe salir.\n');
fprintf('======================================================\n');

while true
    entrada = input('\nPregunta: ','s');
    pregunta = strtrim(string(entrada));
    if strlength(pregunta) == 0
        continue;
    end
    if any(strcmpi(pregunta,["salir","exit","quit"]))
        fprintf('Finalizando el RAG.\n');
        break;
    end
    try
        %% 4.1 LIMPIAR Y CODIFICAR LA PREGUNTA
        preguntaLimpia = limpiarPreguntasParaBusqueda(pregunta);
        vecPreguntaBruto = generarEmbeddings(net,tokenizer,preguntaLimpia, ...
            maxTokensBERT,1,usarGPU);
        vecPregunta = vecPreguntaBruto - mediaGlobal;
        vecPregunta = vecPregunta ./ max(vecnorm(vecPregunta,2,1),eps);
        
        %% 4.2 RECUPERAR LOS CHUNKS MAS RELEVANTES
        simSemantica = embeddings.' * vecPregunta;
        simTokens = puntuarTokensBERT(tokenizer,preguntaLimpia, ...
            chunks,secuenciasTokensChunk,idfTokens);
        puntuacion = pesoSemanticoBERT * simSemantica + ...
            pesoTokensBERT * simTokens;
        [valores,orden] = sort(puntuacion,'descend');
        indicesRecuperados = orden(1:k);
        idsChunks = indicesRecuperados(:).';
        contexto = cell(k,1);
        for j = 1:k
            idx = indicesRecuperados(j);
            contexto{j} = char(chunks(idx));
        end
        
        %% 4.3 GENERAR RESPUESTA CON OLLAMA
        fprintf('Generando respuesta con %s...\n',modeloOllama);
        respuesta = generarRespuestaOllamaLocal(pregunta,contexto,idsChunks, ...
            modeloOllama,urlOllama,tiempoEsperaOllama);
        
        %% 4.4 MOSTRAR RESPUESTA Y FUENTES VERIFICADAS
        fprintf('\n======================================================\n');
        fprintf('PREGUNTA: %s\n',char(pregunta));
        fprintf('RESPUESTA:\n%s\n',respuesta);
        fprintf('\nFUENTES UTILIZADAS (asignadas por MATLAB):\n');
        for j = 1:k
            fprintf('  [Chunk #%d | Puntuacion: %.4f]\n', ...
                idsChunks(j),valores(j));
        end
        fprintf('======================================================\n');
    catch ME
        fprintf(2,'\nERROR al procesar la consulta:\n');
        fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
    end
end

%% ========================================================================
%% FUNCIONES LOCALES
%% ========================================================================
function respuesta = generarRespuestaOllamaLocal(pregunta,contexto,idsChunks,modelo,url,timeout)
    bloques = cell(numel(contexto),1);
    for i = 1:numel(contexto)
        bloques{i} = sprintf('[CHUNK #%d]\n%s', idsChunks(i), contexto{i});
    end
    textoContexto = strjoin(bloques, sprintf('\n\n'));
    
    instrucciones = {
        'Eres un asistente especializado en MATLAB que responde preguntas basándose en fragmentos de un manual técnico.'
        'Debes responder siempre en español y utilizar exclusivamente la información proporcionada en el CONTEXTO.'
        'El CONTEXTO contiene fragmentos de un documento, no necesariamente el documento completo. No presupongas que conoces información que no aparece en esos fragmentos.'
        'No inventes ni completes información utilizando conocimientos externos. No inventes funciones, sintaxis, argumentos, resultados ni ejemplos.'
        'Puedes reorganizar, resumir y explicar la información para que resulte más clara, pero sin modificar su significado técnico.'
        'Si el contexto no contiene información suficiente para responder, indícalo expresamente. Es preferible reconocer que falta información antes que proporcionar una respuesta especulativa.'
        'Si incluyes código MATLAB, asegúrate de que su sintaxis y su funcionamiento estén respaldados por el contexto. No añadas ejemplos de código por iniciativa propia si no están justificados.'
        'No inventes identificadores, números ni referencias a fragmentos. No incluyas citas ni referencias a chunks en tu respuesta.'
        'Responde de forma clara, técnica, concisa y estructurada.'
    };
    systemPrompt = strjoin(instrucciones, ' ');
    userPrompt = sprintf(['PREGUNTA:\n%s\n\n' ...
        'FRAGMENTOS RECUPERADOS DEL MANUAL:\n%s\n\n' ...
        'Responde a la pregunta utilizando estos fragmentos.'], ...
        char(pregunta),textoContexto);
        
    mensajes = struct('role',{},'content',{});
    mensajes(1).role = 'system';
    mensajes(1).content = systemPrompt;
    mensajes(2).role = 'user';
    mensajes(2).content = userPrompt;
    
    solicitud = struct();
    solicitud.model = modelo;
    solicitud.messages = mensajes;
    solicitud.stream = false;
    solicitud.options = struct('temperature',0.2);
    
    opciones = weboptions('MediaType','application/json','Timeout',timeout);
    try
        resultado = webwrite(url,solicitud,opciones);
    catch ME
        error('Fallo al solicitar la respuesta a Ollama: %s',ME.message);
    end
    if ~isfield(resultado,'message') || ~isfield(resultado.message,'content')
        error('Ollama ha devuelto una respuesta con un formato inesperado.');
    end
    respuesta = char(string(resultado.message.content));
    respuesta = regexprep(respuesta,'(?i)\[?\s*chunk\s*#\s*\d+\s*\]?','');
    respuesta = strtrim(respuesta);
    if isempty(respuesta)
        respuesta = 'No se ha recibido una respuesta util del modelo.';
    end
end

function preguntasLimpias = limpiarPreguntasParaBusqueda(preguntas)
    relleno = [ ...
        "¿", "?", "¡", "!", ",", ...
        "cómo", "como", "qué", "que", "cuál", "cual", "para", "sirve", "sirven", ...
        "hay", "entre", "son", "es", "se", "en", "el", "la", "los", "las", ...
        "un", "una", "unos", "unas", "de", "del", "al", "con", "por", "su", "sus", ...
        "o", "y", "a", "matlab" ];
    preguntasLimpias = string(preguntas);
    for i = 1:numel(preguntasLimpias)
        q = lower(preguntasLimpias(i));
        q = replace(q,["¿","?","¡","!",","]," ");
        palabras = split(strtrim(q));
        palabras = palabras(strlength(palabras)>0);
        mantener = ~ismember(palabras,relleno);
        if any(mantener)
            preguntasLimpias(i) = join(palabras(mantener)," ");
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
            a = [a(1:maxTokens-1),tokenizer.SeparatorCode];
            b = [b(1:maxTokens-1),b(end)];
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

function M = generarEmbeddings(net,tokenizer,textos,maxTokens,lote,usarGPU)
    textos = string(textos(:));
    n = numel(textos);
    M = [];
    for primero = 1:lote:n
        ult = min(n,primero+lote-1);
        [ids,masc,seg] = codificarLote(tokenizer,textos(primero:ult),maxTokens);
        entradas = entradasBERT(net,ids,masc,seg,usarGPU);
        Y = predict(net,entradas{:});
        dimT = finddim(Y,'T');
        Xmask = dlarray(masc,'CTB');
        if usarGPU
            Xmask = gpuArray(Xmask);
        end
        Yraw = stripdims(Y);
        Mraw = stripdims(Xmask);
        sumaTokens = sum(Yraw.*Mraw,dimT);
        numTokens = max(sum(Mraw,dimT),1e-8);
        media = sumaTokens./numTokens;
        D = size(Yraw,1);
        E = reshape(media,D,[]);
        E = gather(extractdata(E));
        E = double(E);
        E = E./max(vecnorm(E,2,1),eps);
        if isempty(M)
            M = zeros(D,n,'double');
        end
        M(:,primero:ult) = E;
    end
end

function [secuenciasChunk,idfTokens] = indexarSecuenciasBERT(tokenizer,chunks)
    N = numel(chunks);
    codigos = encode(tokenizer,lower(chunks));
    secuenciasChunk = cell(N,1);
    especiales = double([tokenizer.PaddingCode,tokenizer.StartCode,tokenizer.SeparatorCode]);
    maxId = 150000;
    df = zeros(maxId,1);
    for i = 1:N
        seq = double(codigos{i}(:).');
        seq = seq(~ismember(seq,especiales));
        secuenciasChunk{i} = seq;
        u = unique(seq);
        u = u(u>0 & u<=maxId);
        df(u) = df(u)+1;
    end
    idfBase = log((N+1)./(df+1));
    idfTokens = idfBase.^2;
end

function scores = puntuarTokensBERT(tokenizer,consultaLimpia,chunks,secuenciasChunk,idfTokens)
    N = numel(secuenciasChunk);
    scores = zeros(N,1);
    especiales = double([tokenizer.PaddingCode,tokenizer.StartCode,tokenizer.SeparatorCode]);
    terminos = unique(split(strtrim(lower(consultaLimpia))),'stable');
    terminos = terminos(strlength(terminos)>0);
    if isempty(terminos)
        return;
    end
    numTerm = numel(terminos);
    seqTerminos = cell(numTerm,1);
    pesoTermino = zeros(numTerm,1);
    codTerm = encode(tokenizer,terminos);
    for t = 1:numTerm
        seqT = double(codTerm{t}(:).');
        seqT = seqT(~ismember(seqT,especiales));
        seqTerminos{t} = seqT;
        idsValidos = seqT(seqT>0 & seqT<=numel(idfTokens));
        if ~isempty(idsValidos)
            pesoTermino(t) = max(idfTokens(idsValidos));
        end
    end
    pesoTotal = sum(pesoTermino)+eps;
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
            acierto = contieneSecuencia(seqC,seqT);
            if acierto && strlength(terminos(t))>=2 && ...
                    isletter(char(extractBetween(terminos(t),1,1)))
                patron = "\<"+regexptranslate('escape',terminos(t))+"\>";
                acierto = ~isempty(regexp(chunksLower(i),patron,'once'));
            end
            if acierto
                puntosChunk = puntosChunk+pesoTermino(t);
                terminosAcertados = terminosAcertados+1;
            end
        end
        proporcion = puntosChunk/pesoTotal;
        bonus = 1+0.12*max(0,terminosAcertados-1);
        scores(i) = min(1.0,proporcion*bonus);
    end
end

function encontrado = contieneSecuencia(secuencia,patron)
    secuencia = double(secuencia(:).');
    patron = double(patron(:).');
    encontrado = false;
    n = numel(patron);
    m = numel(secuencia);
    if n==0 || n>m
        return;
    end
    for i = 1:(m-n+1)
        if all(secuencia(i:i+n-1)==patron)
            encontrado = true;
            return;
        end
    end
end