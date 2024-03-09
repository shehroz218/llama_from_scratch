from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
import tqdm

from model import ModelArgs, Transformer

class LLaMa:

    def __init__(self,
                 model: Transformer,
                 tokenizer: SentencePieceProcessor,
                 model_args: ModelArgs):
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod 
    def build(checkpoints_dir: str,
              tokenizer_path: str,
              load_model: bool,
              max_seq_len: int,
              max_batch_size: int,
              device: str):
    
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoints_dir}"
            chk_path = checkpoints[0]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu") #device)
            print(f"Loaded model in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / 'params.json', 'r') as f:
            print(f)
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        ) 

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            # torch.set_default_tensor_type(torch.cuda.HalfTensor)
            # TORCH.SET_DEFAULT_TENSOR_TYPE
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.half)
        else:
            torch.set_default_dtype(torch.Tensor.bfloat16)

        model = Transformer(model_args).to(device)
        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f} seconds")


        return LLaMa(model, tokenizer, model_args)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = LLaMa.build(
                       checkpoints_dir="llama-2-7b/",
                       tokenizer_path="tokenizer.model",
                       load_model=True,
                       max_seq_len=1024,
                       max_batch_size=3,
                    #    device = 'cpu'
                       device=device
                       )
    # print(model)
    # print(model.model)
    # print(model.tokenizer)
    # print(model.model_args)
