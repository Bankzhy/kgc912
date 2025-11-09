from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

max_source_length = 256
max_target_length = 32
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")

def preprocess_function(examples):
    inputs = ["summarize: " + code for code in examples["code"]]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True
    )

    # Tokenize targets
    labels = tokenizer(
        examples["label"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')

    # for index in range(0, len(dataset)):
    #     if index >1000:
    #         break
    #     data = dataset[index]
    #     func_name = data['func_name']
    #     func_code_string = data['func_code_string']
    #     func_code_string_l = func_code_string.split("\n")
    #     max_line = len(func_code_string_l)-2
    #     if max_line <= 0:
    #         max_line = 0
    #     func_code_string_l = func_code_string_l[1:max_line]
    #     fix_func_code_string = '\n'.join(func_code_string_l)
    #     examples["code"].append(fix_func_code_string)
    #     examples["label"].append(func_name)
    #     # print(data)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir="./codet5-mnf",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    trainer.save_model("./codet5-mnf")
    tokenizer.save_pretrained("./codet5-mnf")


if __name__ == '__main__':
    train()