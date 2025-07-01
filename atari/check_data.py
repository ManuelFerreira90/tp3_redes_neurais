import numpy as np
import os

# --- Configure o caminho para o arquivo que queremos checar ---
# Ajuste o nome do jogo se for diferente de 'Breakout'
GAME = 'Breakout' 
# O número do arquivo que vamos checar (pode ser qualquer um de 0 a 199)
BUFFER_NUM = 0 

# Monta o caminho completo para o arquivo .npz
# ATENÇÃO: Verifique se este caminho corresponde exatamente à sua estrutura de pastas e nome de arquivo
file_name = f'{GAME}NoFrameskip-v4_{BUFFER_NUM}.npz'
file_path = os.path.join('./dados_atari', GAME, file_name) 
# ----------------------------------------------------------------

print(f"Tentando inspecionar o arquivo: {file_path}")

try:
    # Tenta carregar o arquivo .npz
    data = np.load(file_path)
    
    # Se conseguir carregar, lista todas as 'chaves' (tipos de dados) dentro dele
    keys = list(data.keys())
    print(f"Arquivo carregado com sucesso! Chaves encontradas: {keys}")
    
    # Se houver chaves, verifica o tamanho (número de transições) de algumas delas
    if 'action' in keys:
        print(f"Shape do array 'action': {data['action'].shape}")
    if 'observation' in keys:
        print(f"Shape do array 'observation': {data['observation'].shape}")

except Exception as e:
    print(f"\nOcorreu um erro ao tentar ler o arquivo: {e}")
    print("Isso indica que o arquivo está corrompido ou o caminho está incorreto.")