import sys
import os
from common.bart import BartForClassificationAndGeneration

curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
sys.path.append('../..')
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
import argparse
import logging
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, AutoModelForMaskedLM, RobertaForTokenClassification, \
    RobertaForMaskedLM, AutoModel, RobertaModel, T5ForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, AutoModelForMaskedLM)
from common import enums
from common.data_collator_kg import collate_fn
logger = logging.getLogger(__name__)
from common.vocab import load_vocab, init_vocab, Vocab
from common.dataset import init_dataset
from common.general import human_format, count_params, layer_wise_parameters

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataloader, model):
    """ Train the model """

    # build dataloader
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 10
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (source_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels) = [x.to(args.device) for x in batch.values()]
            model.train()
            loss, logits = model(source_ids, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, eval_dataloader, eval_when_training=False):
    # build dataloader
    # eval_dataset = T5TextDataset(tokenizer, args, file_path=args.eval_data_file)
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (source_ids, attention_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(source_ids, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, eval_dataloader, best_threshold=0):
    # build dataloader
    # eval_dataset = T5TextDataset(tokenizer, args, file_path=args.test_data_file)
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (source_ids, attention_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(source_ids, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold


    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

    # with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    #     for example, pred in zip(eval_dataset.examples, y_preds):
    #         if pred:
    #             f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
    #         else:
    #             f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')


def run():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/dataset/small10/train.txt", type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="saved_models", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task", default="clone", type=str, required=False,
                        help="The pretrain task.")

    ## Other parameters
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_data_file", default="dataset/dataset/small10/valid.txt", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="dataset/dataset/small10/test.txt", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")

    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.dataset_save_dir = "data/saved"
    args.train_subset_ratio = None
    args.device = device
    args.code_vocab_name = "code"
    args.st_vocab_name = "st"
    args.nl_vocab_name = "nl"
    args.no_ast = False
    args.no_nl = False


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)

    args.config_name = r"H:\research\clone\examples\codet5-base"
    args.tokenizer_name = r"H:\research\clone\examples\codet5-base"
    args.model_name_or_path = r"H:\research\clone\examples\codet5-base"
    args.n_layer = 12
    args.d_ff = 3072
    args.n_head = 12
    args.d_model = 768
    args.dropout = 0.3
    args.max_nl_len = 64
    args.max_ast_len = 32
    args.max_code_len = 256
    args.beam_width = 5

    trained_vocab = "../pretrain/vocabs"
    trained_model = None

    # config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    #
    # config.num_labels = 1
    # tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)

    # config = RobertaConfig.from_pretrained("Salesforce/codet5-base")
    #
    # config.num_labels = 1
    # tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    # model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base", config=config)
    #
    # model=T5CloneModel(model, config, tokenizer,args)
    # trained_model = r""
    # config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
    # model = BartForClassificationAndGeneration.from_pretrained(trained_model, config=config, use_safetensors=True)
    #
    # logger.info("Training/evaluation parameters %s", args)

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    only_test = False
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    splits = ['test'] if only_test else ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = init_dataset(args=args,
                                       task=enums.TASK_CLONE,
                                       split=split)
        logger.info(f'The size of {split} set: {len(datasets[split])}')
    if args.train_subset_ratio and 'train' in datasets:
        datasets['train'] = datasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(datasets['train'])))

    logger.info('Datasets loaded successfully')


    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab is not None:
        logger.info('Loading vocabularies from files')
        code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
        st_vocab = load_vocab(vocab_root=trained_vocab, name=args.st_vocab_name)
        nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)


    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(trained_model,config=config, use_safetensors=True)
    else:
        logger.info('Building the model')
        config = BartConfig(vocab_size=len(code_vocab) + len(nl_vocab) + len(st_vocab),
                            max_position_embeddings=1024,
                            encoder_layers=args.n_layer,
                            encoder_ffn_dim=args.d_ff,
                            encoder_attention_heads=args.n_head,
                            decoder_layers=args.n_layer,
                            decoder_ffn_dim=args.d_ff,
                            decoder_attention_heads=args.n_head,
                            activation_function='gelu',
                            d_model=args.d_model,
                            dropout=args.dropout,
                            use_cache=True,
                            pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                            bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            is_encoder_decoder=True,
                            decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            max_length=args.max_nl_len,
                            min_length=1,
                            num_beams=args.beam_width,
                            num_labels=2)
        model = BartForClassificationAndGeneration(config)
    model.set_model_mode(enums.MODEL_MODE_CLS)
    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # Training
    if args.do_train:
        train_dataloader = DataLoader(dataset=datasets["train"],
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      collate_fn=lambda batch: collate_fn(batch,
                                                                          args=args,
                                                                          task=enums.TASK_CLONE,
                                                                          code_vocab=code_vocab,
                                                                          nl_vocab=nl_vocab,
                                                                          ast_vocab=st_vocab))
        train(args, train_dataloader, model)


    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, '{}'.format("model.bin"))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        eval_dataloader = DataLoader(dataset=datasets["valid"],
                                     batch_size=args.train_batch_size,
                                     shuffle=True,
                                     collate_fn=lambda batch: collate_fn(batch,
                                                                         args=args,
                                                                         task=enums.TASK_CLONE,
                                                                         code_vocab=code_vocab,
                                                                         nl_vocab=nl_vocab,
                                                                         ast_vocab=st_vocab))
        result = evaluate(args, model, eval_dataloader)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, '{}'.format("model.bin"))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test_dataloader = DataLoader(dataset=datasets["test"],
                                     batch_size=args.train_batch_size,
                                     shuffle=True,
                                     collate_fn=lambda batch: collate_fn(batch,
                                                                         args=args,
                                                                         task=enums.TASK_CLONE,
                                                                         code_vocab=code_vocab,
                                                                         nl_vocab=nl_vocab,
                                                                         ast_vocab=st_vocab))
        test(args, model, test_dataloader, best_threshold=0.5)

    return results


if __name__ == '__main__':
    run()