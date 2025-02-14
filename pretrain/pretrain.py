import os
import torch
import logging
import argparse
import sys
curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
sys.path.append('../..')
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)
from transformers.trainer_utils import get_last_checkpoint

import enums
from args import add_args
from dataset import init_dataset
from trainer import CodeTrainer, CodeCLSTrainer
from vocab import load_vocab, init_vocab, Vocab
from bart import BartForClassificationAndGeneration
from general import layer_wise_parameters, human_format, count_params
from transformers import BartConfig, Seq2SeqTrainingArguments, IntervalStrategy, SchedulerType, TrainingArguments
from callbacks import LogStateCallBack
logger = logging.getLogger(__name__)


def pretrain(args):
    tasks = args.pre_train_tasks
    if tasks is None:
        logger.warning("Was specified for pre-training, but got pre-training tasks to None, "
                       "will default to ('mass', 'rlp', 'nlmp')")
        tasks = ['mass', 'rlp', 'nlmp', 'mnp', 'cgp', 'rrlp']
    else:
        supported_tasks = []
        for task in tasks.split(','):
            task = task.strip().lower()
            if task in ['mass', 'rlp', 'nlmp', 'mnp', 'cgp', 'rrlp']:
                supported_tasks.append(task)
            else:
                logger.warning(f'Pre-training task {task} is not supported and will be ignored.')
        tasks = supported_tasks

    trained_model = args.trained_model
    trained_vocab = args.trained_vocab

    if len(os.listdir(trained_vocab)) < 3:
        trained_vocab = None

    if len(os.listdir(trained_model)) == 0:
        trained_model = None

    assert not trained_vocab or isinstance(trained_vocab, str), \
        f'The vocab type is not supported, expect string of vocab dir, got {type(trained_vocab)}'

    logger.info('*' * 100)
    logger.info('Initializing pre-training environments')

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading and parsing datasets')
    dataset = init_dataset(args=args)
    logger.info(f'The size of pre_training set: {len(dataset)}')
    if args.pre_train_subset_ratio:
        dataset = dataset.subset(args.pre_train_subset_ratio)
        logger.info(f'The pre-train dataset is trimmed to subset due to the argument: '
                    f'train_subset_ratio={args.pre_train_subset_ratio}')
        logger.info('The size of trimmed pre-train set: {}'.format(len(dataset)))
    logger.info('Datasets loaded and parsed successfully')
    # dataset = dataset.subset(0.0001)

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab is not None:
        logger.info('Loading vocabularies from files')
        code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
        st_vocab = load_vocab(vocab_root=trained_vocab, name=args.st_vocab_name)
        nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)
    else:
        logger.info('Building vocabularies')
        # code vocab
        code_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                                name=args.code_vocab_name,
                                method=args.code_tokenize_method,
                                vocab_size=args.code_vocab_size,
                                datasets=[dataset.codes],
                                ignore_case=True,
                                save_root=args.vocab_root)
        # nl vocab
        nl_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                              name=args.nl_vocab_name,
                              method=args.nl_tokenize_method,
                              vocab_size=args.nl_vocab_size,
                              datasets=[dataset.nls, dataset.docs] if hasattr(dataset, 'docs') else [dataset.nls],
                              ignore_case=True,
                              save_root=args.vocab_root,
                              index_offset=len(code_vocab))
        # ast vocab
        st_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                               name=args.st_vocab_name,
                               method='word',
                               datasets=[dataset.structures],
                               save_root=args.vocab_root,
                               index_offset=len(code_vocab) + len(nl_vocab))
    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    logger.info(f'The size of ast vocabulary: {len(st_vocab)}')
    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    # logger.info('-' * 100)
    # logger.info('Building model')
    # config = BartConfig(vocab_size=len(code_vocab) + len(st_vocab) + len(nl_vocab),
    #                     max_position_embeddings=512,
    #                     encoder_layers=args.n_layer,
    #                     encoder_ffn_dim=args.d_ff,
    #                     encoder_attention_heads=args.n_head,
    #                     decoder_layers=args.n_layer,
    #                     decoder_ffn_dim=args.d_ff,
    #                     decoder_attention_heads=args.n_head,
    #                     activation_function='gelu',
    #                     d_model=args.d_model,
    #                     dropout=args.dropout,
    #                     use_cache=True,
    #                     pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
    #                     bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
    #                     eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
    #                     is_encoder_decoder=True,
    #                     decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
    #                     forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
    #                     max_length=100,
    #                     min_length=1,
    #                     num_beams=args.beam_width,
    #                     num_labels=2)
    #
    # model = BartForClassificationAndGeneration(config)
    config = BartConfig.from_json_file(os.path.join(args.trained_model, 'config.json'))
    model = BartForClassificationAndGeneration.from_pretrained(args.trained_model, config=config, use_safetensors=True)

    # log model statistic
    logger.info('Model trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')
    logger.info('Tasks:')
    print(tasks)
    # --------------------------------------------------
    # pre-train
    # --------------------------------------------------
    for task in tasks:
        logger.info('-' * 100)
        logger.info(f'Pre-training task: {task.upper()}')

        if isinstance(dataset, torch.utils.data.Subset):
            dataset.dataset.set_task(task)
        else:
            dataset.set_task(task)

        if task == enums.TASK_RLP:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     # auto_find_batch_size=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=10,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.NO,
                                                     save_total_limit=3,
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            print("Current batch size:", training_args.per_device_train_batch_size)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  st_vocab=st_vocab,
                                  nl_vocab=nl_vocab,
                                  task=task,
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mass_result = trainer.train()

            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_MASS:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     # auto_find_batch_size=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.NO,
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            print("Current batch size:", training_args.per_device_train_batch_size)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  st_vocab=st_vocab,
                                  nl_vocab=nl_vocab,
                                  task=task,
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mass_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_NLP or task == enums.TASK_NLMP or task == enums.TASK_MNP:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     # auto_find_batch_size=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.NO,
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            print("Current batch size:", training_args.per_device_train_batch_size)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  st_vocab=st_vocab,
                                  nl_vocab=nl_vocab,
                                  task=task,
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mass_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_CGP:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_CLS)
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = TrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                              overwrite_output_dir=True,
                                              do_train=True,
                                              per_device_train_batch_size=args.batch_size,
                                              gradient_accumulation_steps=1,
                                              learning_rate=args.learning_rate,
                                              weight_decay=args.lr_decay_rate,
                                              max_grad_norm=args.grad_clipping_norm,
                                              num_train_epochs=args.n_epoch,
                                              lr_scheduler_type=SchedulerType.LINEAR,
                                              warmup_steps=args.warmup_steps,
                                              logging_dir=os.path.join(args.tensor_board_root, task),
                                              logging_strategy=IntervalStrategy.STEPS,
                                              logging_steps=args.logging_steps,
                                              save_strategy=IntervalStrategy.NO,
                                              seed=args.random_seed,
                                              fp16=args.fp16,
                                              dataloader_drop_last=False,
                                              run_name=args.model_name,
                                              load_best_model_at_end=True,
                                              ignore_data_skip=False,
                                              label_smoothing_factor=args.label_smoothing,
                                              report_to=['tensorboard'],
                                              dataloader_pin_memory=True)
            trainer = CodeCLSTrainer(main_args=args,
                                     code_vocab=code_vocab,
                                     st_vocab=st_vocab,
                                     nl_vocab=nl_vocab,
                                     task=task,
                                     model=model,
                                     args=training_args,
                                     data_collator=None,
                                     train_dataset=dataset,
                                     tokenizer=nl_vocab,
                                     model_init=None,
                                     compute_metrics=None,
                                     callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            cap_result = trainer.train()
            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

        elif task == enums.TASK_RRLP:
            # set model mode
            logger.info('-' * 100)
            model.set_model_mode(enums.MODEL_MODE_GEN)
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, task),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     # auto_find_batch_size=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, task),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.logging_steps,
                                                     save_strategy=IntervalStrategy.NO,
                                                     save_total_limit=3,
                                                     seed=args.random_seed,
                                                     fp16=args.fp16,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     report_to=['tensorboard'],
                                                     dataloader_pin_memory=True)
            print("Current batch size:", training_args.per_device_train_batch_size)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  st_vocab=st_vocab,
                                  nl_vocab=nl_vocab,
                                  task=task,
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info(f'Start pre-training task: {task}')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mass_result = trainer.train()

            logger.info(f'Pre-training task {task} finished')
            trainer.save_model(os.path.join(args.model_root, task))

    logger.info('Pre-training finished')

    return model, (code_vocab, st_vocab, nl_vocab)

def run_pretrain():
    # --------------------------------------------------
    # load args
    # --------------------------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])
    add_args(parser)
    main_args = parser.parse_args()
    pretrain(main_args)



if __name__ == '__main__':
    run_pretrain()