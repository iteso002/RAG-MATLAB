%% 01_preparar_BERT_multilingue_R2024.m
% MATLAB R2024a / R2024b. Solo MATLAB, sin Python.
% Preparacion de BERT-Base Multilingual Cased (mBERT).
% Requiere Text Analytics Toolbox y Deep Learning Toolbox.
% El complemento se instala desde MathWorks: puede requerir iniciar sesion
% y aceptar una licencia. No hay una API publica documentada en R2024
% que garantice la instalacion totalmente desatendida de este paquete.

clear; clc;
fprintf('MATLAB: %s\n', version);
assert(~verLessThan('matlab','24.1'), ...
    'Este ejemplo esta dirigido a MATLAB R2024a/R2024b o posteriores.');

if isempty(ver('textanalytics'))
    error('Falta Text Analytics Toolbox. Instalalo con tu licencia MATLAB.');
end
if isempty(ver('nnet'))
    error('Falta Deep Learning Toolbox. Instalalo con tu licencia MATLAB.');
end

fprintf('Cargando BERT-Base Multilingual Cased...\n');
try
    [net,tokenizer] = bert(Model="multilingual");
catch ME
    fprintf(2, '\nNo se ha podido cargar BERT multilingue.\n');
    fprintf(2, 'Detalle de MATLAB: %s\n\n', ME.message);
    fprintf('Se abrira el complemento oficial de MathWorks.\n');
    fprintf('Pulsa Install, inicia sesion/acepta licencia si se solicita,\n');
    fprintf('y ejecuta de nuevo este mismo archivo.\n\n');
    web('https://www.mathworks.com/add-ons/BERT_MULTILINGUAL/','-browser');
    error('Instala el complemento de BERT multilingue y vuelve a ejecutar.');
end

fprintf('BERT cargado correctamente.\n');
fprintf('Contexto maximo del tokenizador: %d tokens.\n', tokenizer.ContextSize);
fprintf('Entradas de la red: %s\n', strjoin(string(net.InputNames), ', '));

% Guardado local para usarlo sin volver a descargarlo.
% El archivo puede superar varios cientos de MB.
archivo = fullfile(fileparts(mfilename('fullpath')), ...
    'bert_multilingue_base.mat');
save(archivo,'net','tokenizer','-v7.3');
fprintf('Modelo guardado en:\n%s\n', archivo);
