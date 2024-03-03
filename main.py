# from collator import DataCollatorSpeechSeq2SeqWithPadding
from load_data import LoadData
from processing import Preprocess

data = LoadData()
dataset = data.download_dataset()
preprocessor = Preprocess(dataset)
prepared_test_dataset = dataset["test"].map(preprocessor.prepare_dataset)
tokenizer = preprocessor.tokenizer()

input_str = prepared_test_dataset["test"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
# print(prepared_test_dataset)