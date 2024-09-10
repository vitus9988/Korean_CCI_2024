
import json
import tqdm
from peft import PeftModel
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from src.data import CustomDataset


def main(adapter_path):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    
    peft_model_id = adapter_path
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        quantization_config=quant_config,
        device_map='cuda:0',
    )

    model = PeftModel.from_pretrained(model, peft_model_id)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = CustomDataset("resource/data/test.json", tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("resource/data/test.json", "r") as f:
        result = json.load(f)

    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(dataset))):
            inp, _ = dataset[idx]
            outputs = model(
                inp.to('cuda:0').unsqueeze(0)
            )
            logits = outputs.logits[:,-1].flatten()
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer.vocab['A']],
                            logits[tokenizer.vocab['B']],
                            logits[tokenizer.vocab['C']],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            result[idx]["output"] = answer_dict[numpy.argmax(probs)]

    resultfilename = adapter_path.split('/')[-1]
    with open(f"../answer/{resultfilename}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    from pathlib import Path
    import time
    directory = '../adapters' 
    adapters_path = [str(subdir) for subdir in Path(directory).iterdir() if subdir.is_dir()]
    for adapter in adapters_path:
        main(adapter)
        time.sleep(1)



