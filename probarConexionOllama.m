%% probarConexionOllama.m
% Comprueba que MATLAB puede comunicarse con Ollama y muestra los modelos
% instalados. No entrena BERT ni modifica los archivos del RAG.

clear; clc;

urlBase = 'http://localhost:11434';

try
    opcionesHTTP = weboptions('Timeout', 30);
    datos = webread([urlBase '/api/tags'], opcionesHTTP);

    fprintf('\nConexión con Ollama correcta.\n');
    fprintf('Modelos instalados:\n\n');

    if isfield(datos, 'models') && ~isempty(datos.models)
        for k = 1:numel(datos.models)
            fprintf('  %s\n', datos.models(k).name);
        end
    else
        fprintf('  No hay modelos instalados.\n');
    end
catch ME
    fprintf(2, '\nNo se ha podido conectar con Ollama.\n');
    fprintf(2, 'Comprueba que Ollama está iniciado y que funciona en %s.\n', urlBase);
    fprintf(2, 'Detalle: %s\n', ME.message);
end
