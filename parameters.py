import argparse
import os


def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--dataset', default='cub200', type=str,
                        help='Dataset to use.')
    parser.add_argument('--use_tv_split', action='store_true',
                        help='Flag. If set, splits training set into a train/validation setup following --tv_split_perc.')
    parser.add_argument('--tv_split_by_samples', action='store_true',
                        help='Whether to split train/validation sets by splitting per class or between classes.')
    parser.add_argument('--tv_split_perc', default=0,  type=float,
                        help='Percentage with which the training dataset is split into training/validation.')
    parser.add_argument('--completed', action='store_true',
                        help='Flag. If set, denotes that the training process has been completed.')
    parser.add_argument('--no_train_metrics', action='store_true',
                        help='Flag. If set, no training metrics are computed and logged.')
    parser.add_argument('--no_test_metrics', action='store_true',
                        help='Flag. If set, no test metrics are computed and logged.')
    #
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'e_recall@10', 'nmi', 'f1', 'mAP_1000'], type=str,
                        help='Metrics to evaluate performance by.')
    parser.add_argument('--evaltypes', nargs='+', default=['embeds'], type=str,
                        help='The network may produce multiple embeddings (ModuleDict). If the key is listed here, the entry will be evaluated on the evaluation metrics. Note: One may use Combined_embed1_embed2_..._embedn-w1-w1-...-wn to compute evaluation metrics on weighted (normalized) combinations.')
    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'], type=str,
                        help='Improvement in these metrics will trigger checkpointing.')
    parser.add_argument('--store_improvements', action='store_true',
                        help='If set, will store checkpoints whenever the storage metric improves.')
    #
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--savename', default='group_plus_seed',   type=str,
                        help='Appendix to save folder name if any special information is to be included.')
    parser.add_argument('--source_path', default=os.getcwd()+'/../../Datasets',   type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default=os.getcwd()+'/Training_Results', type=str,
                        help='Where to save everything.')
    ### General Optimization Parameters
    parser.add_argument('--lr',  default=0.00001, type=float,
                        help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr', default=-1, type=float,
                        help='Learning Rate for mlp head parameters. If -1, uses the same base lr.')
    parser.add_argument('--n_epochs', default=150, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--kernels', default=6, type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs', default=112 , type=int,
                        help='Mini-Batchsize to use.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--gamma', default=0.3, type=float,
                        help='Learning rate reduction after --tau epochs.')
    parser.add_argument('--decay', default=0.0004, type=float,
                        help='Weight decay for optimizer.')
    parser.add_argument('--tau', default=[10000],  nargs='+',type=int,
                        help='Stepsize before reducing learning rate.')
    parser.add_argument('--augmentation', default='base', type=str,
                        help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--warmup', default=0, type=int,
                        help='Appendix to save folder name if any special information is to be included.')
    #
    parser.add_argument('--evaluate_on_cpu', action='store_true',
                        help='Flag. If set, computed evaluation metrics on CPU instead of GPU.')
    parser.add_argument('--internal_split', default=1, type=float,
                        help='Split parameter used for meta-learning extensions.')
    #
    parser.add_argument('--optim', default='adam', type=str,
                        help='Optimizer to use.')
    parser.add_argument('--loss', default='margin', type=str,
                        help='Trainin objective to use. See folder <criteria> for available methods.')
    parser.add_argument('--batch_mining', default='distance', type=str,
                        help='Batchmining method to use. See folder <batchminer> for available methods.')
    #
    parser.add_argument('--embed_dim', default=128, type=int,
                        help='Embedding dimensionality of the network. Note: dim=128 or 64 is used in most papers.')
    parser.add_argument('--arch', default='resnet50_frozen_normalize',  type=str,
                        help='Underlying network architecture. Frozen denotes that exisiting pretrained batchnorm layers are frozen, and normalize denotes normalization of the output embedding.')
    parser.add_argument('--not_pretrained', action='store_true',
                        help='Flag. If set, does not initialize the backbone network with ImageNet pretraining.')
    parser.add_argument('--use_float16', action='store_true',
                        help='Flag. If set, uses float16-inputs.')
    return parser


def wandb_parameters(parser):
    """
    Parameters for Weights & Biases logging.
    """
    parser.add_argument('--log_online', action='store_true',
                        help='Flag. If set, logs key data to W&B servers.')
    parser.add_argument('--wandb_key', default='<your_wandb_key>', type=str,
                        help='W&B account key.')
    parser.add_argument('--project', default='DiVA_Sample_Runs', type=str,
                        help='W&B Project name.')
    parser.add_argument('--group', default='Sample_Run', type=str,
                        help='W&B Group name - allows you to group multiple seeds within the same group.')
    return parser


def language_guidance_parameters(parser):
    """
    Parameters for Language Guidance.
    """
    parser.add_argument('--language_distill_w', default=0, type=float,
                        help='Language guidance weight.')
    parser.add_argument('--language_model', default='clip', type=str,
                        help='Pretrained language model to use.')
    parser.add_argument('--language_delay', default=0, type=int,
                        help='Number of iterations after which language guidance is activated.')
    parser.add_argument('--language_pseudoclass', action='store_true',
                        help='Flag. If set, uses ImageNet pseudoclass as language tokens.')
    parser.add_argument('--language_pseudoclass_topk', default=5, type=int,
                        help='Number of pseudoclass tokens to use per sample/class. Higher values create longer and more unique pseudoclass lists, but introduce more noise.')
    parser.add_argument('--language_shift', default=1, type=float,
                        help='Shift for language similarity distribution.')
    parser.add_argument('--language_distill_dir', default='backward', type=str,
                        help='Language-to-image distillation direction.')
    parser.add_argument('--language_temp', default=1, type=float,
                        help='Temperature for KL-Distillation of language-to-image similarities.')
    return parser


def loss_specific_parameters(parser):
    """
    Hyperparameters for various base DML criteria.
    """
    ### Contrastive Loss.
    parser.add_argument('--loss_contrastive_pos_margin', default=0, type=float,
                        help='Positive margin for contrastive pairs.')
    parser.add_argument('--loss_contrastive_neg_margin', default=1, type=float,
                        help='Negative margin for contrastive pairs.')
    ### Triplet-based Losses.
    parser.add_argument('--loss_triplet_margin', default=0.2, type=float,
                        help='Margin for Triplet Loss')
    ### S2SD.
    parser.add_argument('--loss_s2sd_source', default='multisimilarity', type=str,
                        help='DML criterion for the base embedding branch.')
    parser.add_argument('--loss_s2sd_target', default='multisimilarity', type=str,
                        help='DML criterion for the target embedding branches.')
    parser.add_argument('--loss_s2sd_T', default=1, type=float,
                        help='Temperature for the KL-Divergence Distillation.')
    parser.add_argument('--loss_s2sd_w', default=0, type=float,
                        help='Weight of the distillation loss.')
    parser.add_argument('--loss_s2sd_pool_aggr', action='store_true',
                        help='Flag. If set, uses both global max- and average pooling in the target branches.')
    parser.add_argument('--loss_s2sd_target_dims', default=[1024,1536,2048], nargs='+', type=int,
                        help='Defines number and dimensionality of used target branches.')
    parser.add_argument('--loss_s2sd_feat_distill', action='store_true',
                        help='Flag. If set, feature distillation is used.')
    parser.add_argument('--loss_s2sd_feat_w', default=50, type=float,
                        help='Weight of the feature space distillation loss.')
    parser.add_argument('--loss_s2sd_feat_distill_delay', default=1000, type=int,
                        help='Defines the number of training iterations before feature distillation is activated.')
    ### MarginLoss.
    parser.add_argument('--loss_margin_margin', default=0.2, type=float,
                        help='Base Margin parameter in MarginLoss')
    parser.add_argument('--loss_margin_beta_lr', default=0.0005, type=float,
                        help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta', default=1.2, type=float,
                        help='Initial class margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu', default=0, type=float,
                        help='Regularisation value on betas in Margin Loss.')
    parser.add_argument('--loss_margin_beta_constant', action='store_true',
                        help='Flag. If set, class margin values are not trained.')
    ### ProxyAnchor.
    parser.add_argument('--loss_oproxy_mode', default='anchor', type=str,
                        help='Proxy-method: anchor = ProxyAnchor, nca = ProxyNCA.')
    parser.add_argument('--loss_oproxy_lrmulti', default=2000, type=float,
                        help='Learning rate multiplier for proxies.')
    parser.add_argument('--loss_oproxy_pos_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for positive sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_neg_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for negative sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_pos_delta', default=0.1, type=float,
                        help='Threshold for positive sample-proxy similarities')
    parser.add_argument('--loss_oproxy_neg_delta', default=-0.1, type=float,
                        help='Threshold for negative sample-proxy similarities')
    ### NPair.
    parser.add_argument('--loss_npair_l2', default=0.005, type=float,
                        help='L2 weight in NPair. Note: Set to 0.02 in paper, but multiplied with 0.25 in the implementation as well.')
    ### Multisimilary Loss.
    parser.add_argument('--loss_multisimilarity_dim', default=0, type=int,
                        help='Dimension of affinity matrix along which logsumexp is computed.')
    parser.add_argument('--loss_multisimilarity_pos_weight', default=2, type=float,
                        help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight', default=40, type=float,
                        help='Weighting on negative similarities.')
    parser.add_argument('--loss_multisimilarity_margin', default=0.1, type=float,
                        help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_pos_thresh', default=0.5, type=float,
                        help='Theshold on positive similarities (same class).')
    parser.add_argument('--loss_multisimilarity_neg_thresh', default=0.5, type=float,
                        help='Theshold on negative similarities (different class)')
    ### Normalized Softmax Loss.
    parser.add_argument('--loss_softmax_lr', default=0.00001, type=float,
                        help='Learning rate for softmax proxies.')
    parser.add_argument('--loss_softmax_temperature', default=0.05, type=float,
                        help='Temperature for normalized softmax loss.')
    ### ArcFace Loss.
    parser.add_argument('--loss_arcface_lr', default=0.0005, type=float,
                        help='Learning rate for arcface proxies.')
    parser.add_argument('--loss_arcface_angular_margin', default=0.5, type=float,
                        help='Angular margin between proxies.')
    parser.add_argument('--loss_arcface_feature_scale', default=16, type=float,
                        help='Inverted Temperature/scaling factor.')
    return parser


def batchmining_specific_parameters(parser):
    """
    Hyperparameters for various batchmining methods.
    """
    ### Distance-based_Sampling.
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,
                        help='Cutoff distance value below which pairs are ignored.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float,
                        help='Cutoff distance value above which pairs are ignored.')
    ### Spectrum-Regularized Miner.
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float,
                        help='Probability with which label assignments are flipped.')
    ### Semihard Batchmining.
    parser.add_argument('--miner_semihard_margin', default=0.2, type=float,
                        help='Margin value for semihard mining.')
    return parser


def batch_creation_parameters(parser):
    """
    Parameters for batch sampling methods.
    """
    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='Batch-creation method. Default <class_random> ensures that for each class, at least --samples_per_class samples per class are available in each minibatch.')
    parser.add_argument('--data_ssl_set', action='store_true',
                        help='Obsolete. Only relevant for SSL-based extensions.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    return parser
