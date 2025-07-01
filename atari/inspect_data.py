import numpy as np

file_path = 'dataset_breakout.npy'

print(f"Inspecionando o arquivo: {file_path}")

try:
    data = np.load(file_path, allow_pickle=True)

    print(f"\nTipo do dado carregado: {type(data)}")
    print(f"Shape do dado carregado: {data.shape}")
    print(f"Dtype do dado carregado: {data.dtype}")

    if data.shape == (): 
        print("\nO dado é um array de dimensão 0. Acessando seu conteúdo...")
        content = data.item()
        print(f"Tipo do conteúdo: {type(content)}")
        
        if isinstance(content, dict):
            print("O conteúdo é um dicionário. Chaves encontradas:")
            keys = list(content.keys())
            print(keys)
            
            print("\nAnalisando o shape de cada item no dicionário:")
            for key, value in content.items():
                if hasattr(value, 'shape'):
                    print(f"  - Shape de '{key}': {value.shape}")

except Exception as e:
    print(f"\nOcorreu um erro: {e}")