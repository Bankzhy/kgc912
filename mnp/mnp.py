import argparse
import logging
import os
import sys

from transformers import BartConfig, IntervalStrategy, SchedulerType, Seq2SeqTrainingArguments, EarlyStoppingCallback

curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
sys.path.append('../..')
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)

from common.bart import BartForClassificationAndGeneration
from common.callbacks import LogStateCallBack

from common.dataset import KGCodeDataset
from common.trainer import CodeTrainer
from common.vocab import load_vocab
from common import enums
from args import add_args
from sum.eval import bleu, rouge_l
from sum.eval.metrics import avg_ir_metrics, accuracy_for_sequence
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def run_mnp():

    # --------------------------------------------------
    # load args
    # --------------------------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])
    add_args(parser)
    main_args = parser.parse_args()
    args = main_args
    trained_model = args.trained_model
    trained_vocab = args.trained_vocab

    logger.info('Loading vocabularies from files')
    code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
    st_vocab = load_vocab(vocab_root=trained_vocab, name=args.st_vocab_name)
    nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)

    config = BartConfig.from_json_file(os.path.join(args.trained_model, 'config.json'))
    model = BartForClassificationAndGeneration.from_pretrained(args.trained_model, config=config, use_safetensors=True)

    # set model mode
    logger.info('-' * 100)
    model.set_model_mode(enums.MODEL_MODE_GEN)

    train_dataset = KGCodeDataset(args=args, task=enums.TASK_MNP, split="train")
    test_dataset = KGCodeDataset(args=args, task=enums.TASK_MNP, split="test")

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = nl_vocab.decode_batch(preds)
        decoded_labels = nl_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    def compute_test_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    def compute_valid_metrics(eval_preds):
        decoded_labels, decoded_preds = decode_preds(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        return result

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')
    # 尽量不要用IntervalStrategy.EPOCH， 太过频繁影响训练效果，还会曾家训练时间
    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root, enums.TASK_MNP),
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.STEPS,
                                             eval_steps=500,
                                             prediction_loss_only=False,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root, enums.TASK_MNP),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=500,
                                             save_strategy=IntervalStrategy.STEPS,
                                             save_steps=500,
                                             save_total_limit=3,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             metric_for_best_model='bleu',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    trainer = CodeTrainer(main_args=args,
                          code_vocab=code_vocab,
                          st_vocab=st_vocab,
                          nl_vocab=nl_vocab,
                          task=enums.TASK_MNP,
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=train_dataset,
                          eval_dataset=test_dataset,
                          tokenizer=nl_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    logger.info('Running configurations initialized successfully')

    only_test = False
    # --------------------------------------------------
    # train
    # --------------------------------------------------
    if not only_test:
        logger.info('-' * 100)
        # logger.info('loading checkpoint')
        # last_checkpoint = get_last_checkpoint(os.path.join(args.checkpoint_root, enums.TASK_SUMMARIZATION),)
        logger.info('Start training')
        # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        train_result = trainer.train()
        logger.info('Training finished')
        trainer.save_model(args.model_root)
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics(split='train', metrics=metrics)
        trainer.save_metrics(split='train', metrics=metrics)

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Start testing')
    trainer.compute_metrics = compute_test_metrics
    predict_results = trainer.predict(test_dataset=test_dataset,
                                      metric_key_prefix='test',
                                      max_length=args.max_nl_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_MNP}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_MNP}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_MNP}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')


if __name__ == '__main__':
    run_mnp()