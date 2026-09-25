import torch
from src.model import NanoGPT

DEVICE = "cpu"
BLOCK_SIZE = 64

model = NanoGPT(vocab_size=256, block_size=BLOCK_SIZE, dim=128, n_layers=6)

try:
    model.load_state_dict(torch.load("checkpoints/nanogpt_epoch5.pt", map_location=DEVICE))
    print("Loaded epoch5")
except:
    try:
        model.load_state_dict(torch.load("checkpoints/nanogpt_epoch1.pt", map_location=DEVICE))
    except:
        print("Using random weights - train first!")

model.eval()

prompts = [
    "Жил-был мальчик по имени Петя",
    "Тётя Наташа! Возьмите",
    "Ходики показывают"
]

for prompt in prompts:
    x = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(120):
            x_cond = x[:, -BLOCK_SIZE:] if x.size(1) > BLOCK_SIZE else x
            logits, _ = model(x_cond)
            # temperature 0.8
            probs = torch.softmax(logits[0,-1] / 0.8, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            x = torch.cat([x, torch.tensor([[next_id]])], dim=1)

    out = ''.join([chr(c) for c in x[0].tolist()])
    print(f"\n--- Prompt: {prompt} ---")
    print(out)
    print("-" * 40)
