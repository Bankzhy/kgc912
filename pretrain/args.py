import dataclasses
from dataclasses import dataclass, field

@dataclass
class RuntimeArguments:
    trained_vocab: str = field(
        default='vocabs/',
        metadata={'help': 'Directory of trained vocabs'}
    )

    pre_train_tasks: str = field(
        default='mass',
        metadata={'help': 'Pre-training tasks in order, split by commas, '
                          'for example (mass,rlp,nlp)}'}
    )

    trained_model: str = field(
        default=None,
        metadata={'help': 'Directory of trained model'}
    )

    dataset_save_dir: str = field(
        default='data/saved',
        metadata={'help': 'Directory to save and load dataset pickle instance'}
    )

    dataset_root: str = field(
        default='kg_data',
        metadata={'help': 'Directory to save and load dataset pickle instance'}
    )

    pre_train_subset_ratio: float = field(
        default=None,
        metadata={'help': 'Ratio of pre-train subset'}
    )

    vocab_root: str = field(
        default='vocabs',
        metadata={'help': 'Directory to save and load vocab pickle instance'}
    )

    vocab_save_dir: str = field(
        default='vocabs/saved',
        metadata={'help': 'Directory to save and load vocab pickle instance'}
    )

    code_vocab_name: str = field(
        default='code',
        metadata={'help': 'Name of the code vocab'}
    )

    st_vocab_name: str = field(
        default='st',
        metadata={'help': 'Name of the st vocab'}
    )

    nl_vocab_name: str = field(
        default='nl',
        metadata={'help': 'Name of the nl vocab'}
    )

    code_tokenize_method: str = field(
        default='bpe',
        metadata={'help': 'Tokenize method of code',
                  'choices': ['word', 'bpe']}
    )

    nl_tokenize_method: str = field(
        default='bpe',
        metadata={'help': 'Tokenize method of nl',
                  'choices': ['word', 'bpe']}
    )

    pre_train_output_root: str = field(
        default='output/pretrain',
        metadata={'help': 'pretrain model output '}
    )

    model_root: str = field(
        default='output/pretrain/models',
        metadata={'help': 'pretrain model output '}
    )

    tensor_board_root: str = field(
        default='output/runs',
        metadata={'help': 'pretrain model run output '}
    )

    code_vocab_size: int = field(
        default=50000,
        metadata={'help': 'Maximum size of code vocab'}
    )

    nl_vocab_size: int = field(
        default=30000,
        metadata={'help': 'Maximum size of nl vocab'}
    )

    n_layer: int = field(
        default=12,
        metadata={'help': 'Number of layer'}
    )

    d_ff: int = field(
        default=3072,
        metadata={'help': 'Dimension of the feed forward layer'}
    )

    n_head: int = field(
        default=12,
        metadata={'help': 'Number of head of self attention'}
    )

    d_model: int = field(
        default=768,
        metadata={'help': 'Dimension of the model'}
    )

    dropout: float = field(
        default=0.1,
        metadata={'help': 'Dropout probability'}
    )

    random_seed: int = field(
        default=42,
        metadata={'help': 'Specific random seed manually for all operations, 0 to disable'}
    )

    n_epoch: int = field(
        default=30,
        metadata={'help': 'Number of data iterations for training'}
    )

    batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size for training on each device'}
    )

    eval_batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size for evaluation on each device'}
    )

    beam_width: int = field(
        default=5,
        metadata={'help': 'Beam width when using beam decoding, 1 to greedy decode'}
    )

    logging_steps: int = field(
        default=100,
        metadata={'help': 'Log training state every n steps'}
    )

    cuda_visible_devices: str = field(
        default=None,
        metadata={'help': 'Visible cuda devices, string formatted, device number divided by \',\', e.g., \'0, 2\', '
                          '\'None\' will use all'}
    )

    fp16: bool = field(
        default=True,
        metadata={'action': 'store_true',
                  'help': 'Whether to use mixed precision'}
    )

    learning_rate: float = field(
        default=5e-5,
        metadata={'help': 'Learning rate'}
    )

    lr_decay_rate: float = field(
        default=0,
        metadata={'help': 'Decay ratio for learning rate, 0 to disable'}
    )

    early_stop_patience: int = field(
        default=20,
        metadata={'help': 'Stop training if performance does not improve in n epoch, 0 to disable'}
    )

    warmup_steps: int = field(
        default=1000,
        metadata={'help': 'Warmup steps for optimizer, 0 to disable'}
    )

    grad_clipping_norm: float = field(
        default=1.0,
        metadata={'help': 'Gradient clipping norm, 0 to disable'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Gradient accumulation steps, default to 1'}
    )

    label_smoothing: float = field(
        default=0.1,
        metadata={'help': 'Label smoothing ratio, 0 to disable'}
    )

    model_name: str = field(
        default='default_model',
        metadata={'help': 'Name of the model'}
    )

    mass_mask_ratio: float = field(
        default=0.5,
        metadata={'help': 'Ratio between number of masked tokens and number of total tokens, in MASS'}
    )

    max_code_len: int = field(
        default=256,
        metadata={'help': 'Maximum length of code sequence'}
    )

    max_ast_len: int = field(
        default=64,
        metadata={'help': 'Maximum length of ast sequence'}
    )

    max_nl_len: int = field(
        default=64,
        metadata={'help': 'Maximum length of the nl sequence'}
    )

    no_ast: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to eliminate AST from input'}
    )

    no_nl: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to eliminate natural language from input'}
    )



def transfer_arg_name(name):
    return '--' + name.replace('_', '-')

def add_args(parser):
    """Add all arguments to the given parser."""
    for data_container in [RuntimeArguments]:
        group = parser.add_argument_group(data_container.__name__)
        for data_field in dataclasses.fields(data_container):
            if 'action' in data_field.metadata:
                group.add_argument(transfer_arg_name(data_field.name),
                                   default=data_field.default,
                                   **data_field.metadata)
            else:
                group.add_argument(transfer_arg_name(data_field.name),
                                   type=data_field.type,
                                   default=data_field.default,
                                   **data_field.metadata)