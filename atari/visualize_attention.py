# visualize_attention.py (VERSÃO CORRIGIDA)

import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Carregando dados de atenção de 'attention_data.pkl'...")
with open('attention_data.pkl', 'rb') as f:
    viz_data = pickle.load(f)

last_frame = viz_data['state_image'][0, 0, -1, :, :]
attention_weights = viz_data['attention_weights']
mean_attention = np.mean(attention_weights, axis=1).squeeze(0)

attention_vector = mean_attention[-1, :]


attention_on_returns = attention_vector[0::3]
attention_on_states = attention_vector[1::3]
attention_on_actions = attention_vector[2::3]

min_len = min(len(attention_on_returns), len(attention_on_states), len(attention_on_actions))

attention_on_returns = attention_on_returns[:min_len]
attention_on_states = attention_on_states[:min_len]
attention_on_actions = attention_on_actions[:min_len]

timesteps = np.arange(-min_len, 0)


fig, axs = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 2]})

axs[0].imshow(last_frame, cmap='gray')
axs[0].set_title('Frame do Jogo no Momento da Decisão')
axs[0].axis('off')

width = 0.3
axs[1].bar(timesteps - width, attention_on_returns, width, label='Atenção em Retornos (R)', color='g', alpha=0.7)
axs[1].bar(timesteps, attention_on_states, width, label='Atenção em Estados (S)', color='b', alpha=0.7)
axs[1].bar(timesteps + width, attention_on_actions, width, label='Atenção em Ações (A)', color='r', alpha=0.7)

axs[1].set_title('Distribuição da Atenção do Transformer no Passado')
axs[1].set_xlabel('Passos de Tempo Anteriores (t-k)')
axs[1].set_ylabel('Peso da Atenção (Softmax)')
axs[1].legend()
axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)
axs[1].set_xticks(timesteps) 
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('attention_visualization.png')
print("Visualização salva em 'attention_visualization.png'")
plt.show()