from transformers import WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import argparse


class Merger:
    def merge_lora_weights(adapter_model_id, merged_model_id, huggingface_token):
        """
        Merge LoRA weights with a pre-trained Whisper model and upload to a new repository.

        Args:
            adapter_model_id (str): The model ID containing LoRA adapters
            merged_model_id (str): The model ID where the merged model will be saved
            huggingface_token (str): The Hugging Face write token for authentication
        """
        peft_config = PeftConfig.from_pretrained(adapter_model_id)
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, adapter_model_id)
        merged_model = model.merge_and_unload()
        merged_model.train(False)

        # Push to a different repository
        merged_model.push_to_hub(repo_id=merged_model_id, token=huggingface_token)
        print(f"LoRA weights from {adapter_model_id} merged and saved to {merged_model_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with a pre-trained Whisper model and upload to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="The Hugging Face model ID for the LoRA configuration.")
    parser.add_argument("--huggingface_token", type=str, help="The Hugging Face write token for authentication.")
    args = parser.parse_args()

    Merger.merge_lora_weights(args.hf_model_id, args.huggingface_token)

if __name__ == "__main__":
    main()
