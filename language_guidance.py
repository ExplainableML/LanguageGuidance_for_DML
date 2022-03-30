import clip
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


class LanguageGuide(torch.nn.Module):
    def __init__(self,
                 device,
                 language_model='clip',
                 activation_iter=0,
                 language_shift=1,
                 use_pseudoclasses=True,
                 pseudoclass_topk=5,
                 sample_level=False,
                 distill_dir='forward',
                 T=1):
        super(LanguageGuide, self).__init__()

        self.language_model = language_model_select(language_model, device)
        self.activation_iter = activation_iter
        self.language_shift = language_shift
        self.iter_count = 0

        self.use_pseudoclasses = use_pseudoclasses
        self.pseudoclass_topk = pseudoclass_topk
        self.sample_level = sample_level

        self.distill_dir = distill_dir
        self.language_embeds = None

        self.T = T

    def regularize(self, batch, labels, sample_indices=None):
        assert self.language_embeds is not None, 'Please precompute language embeddings first!'

        self.iter_count += 1
        if self.iter_count >= self.activation_iter:
            if isinstance(self.language_embeds, dict):
                language_embeds = torch.stack([
                    self.language_embeds[idx]
                    for idx in labels.detach().cpu().numpy()
                ])
                language_embeds = language_embeds.unsqueeze(1).to(batch.device)
            else:
                indexer = sample_indices if self.sample_level else labels
                language_embeds = self.language_embeds[indexer]

            bsame_labels = (labels.T == labels.view(-1, 1)).to(batch.device).T

            language_sims = self.compute_language_sims(language_embeds)
            language_sims = language_sims.mean(dim=-1)
            language_sims += self.language_shift
            maskval = 1 + self.language_shift

            batch_sims = batch.mm(batch.T)

            if self.distill_dir == 'forward':
                return kl_div(batch_sims,
                              language_sims.detach(),
                              mask=bsame_labels,
                              maskval=maskval,
                              T=self.T)
            else:
                return kl_div(language_sims.detach(),
                              batch_sims,
                              mask=bsame_labels,
                              maskval=maskval,
                              T=self.T)

        return 0.

    def precompute_language_embeds(self,
                                   dataloader,
                                   device,
                                   pseudoclass_generator=None):
        if self.use_pseudoclasses:
            self.classlevel_relabels, self.sample_relabels = relabel(
                pseudoclass_generator,
                dataloader,
                device,
                topk=self.pseudoclass_topk)

            self.language_embeds = reembed_in_language(
                self.language_model, self.classlevel_relabels
                if not self.sample_level else self.sample_relabels, device)

            self.language_embeds = self.language_embeds.to(device)
            if not self.sample_level:
                self.language_embeds = self.language_embeds.permute(1, 0, 2)
            print('Retrieved {} language embeddings!'.format(
                self.language_embeds.shape[0] * self.language_embeds.shape[1]))
        else:
            self.language_embeds = reembed_dict_in_language(
                self.language_model, dataloader.dataset.language_conversion,
                device)
            print('Retrieved {} language embeddings!'.format(
                len(self.language_embeds)))

    def compute_language_sims(self, language_embeds):
        language_sims = torch.einsum(
            'abe,cbe->acb',
            torch.nn.functional.normalize(language_embeds, dim=-1),
            torch.nn.functional.normalize(language_embeds, dim=-1))
        language_sims = language_sims.reshape(*language_sims.shape[:2], -1)
        return language_sims


def kl_div(A, B, mask=None, maskval=0., T=1):
    if mask is not None:
        log_p_A = F.log_softmax(A.masked_fill(mask, maskval) / T, dim=-1)
        p_B = F.softmax(B.masked_fill(mask, maskval) / T, dim=-1)
    else:
        log_p_A = F.log_softmax(A / T, dim=-1)
        p_B = F.softmax(B / T, dim=-1)
    kl_div = F.kl_div(log_p_A, p_B, reduction='sum') * (T**2) / A.shape[0]
    return kl_div


def language_model_select(model, device, primer='a photo of a {}'):
    if model not in ['clip', 'bert', 'roberta_l']:
        raise NotImplementedError(
            'Language model {} not available!'.format(model))
    if model == 'clip':
        return ClipLanguageModel(primer, device)
    if model == 'bert':
        return BertLanguageModel(primer)
    if model == 'roberta_l':
        return RobertaLargeLanguageModel(primer)


class ClipLanguageModel(torch.nn.Module):
    def __init__(self, primer, device):
        super(ClipLanguageModel, self).__init__()
        self.name = 'CLIP-Language'
        self.primer = primer
        self.tokenizer = clip.tokenize
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.out_dim = 512

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens)
        language_embeds = self.model.encode_text(primed_tokens.to(device))
        return language_embeds.type(torch.float32)


class RobertaLargeLanguageModel(torch.nn.Module):
    def __init__(self, primer, **kwargs):
        super(RobertaLargeLanguageModel, self).__init__()
        from transformers import RobertaTokenizer, RobertaModel
        self.name = 'Roberta-Large'
        self.primer = primer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large')
        self.out_dim = 1024

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True).to(device)
        language_embeds = self.model(**primed_tokens).pooler_output
        return language_embeds.type(torch.float32)


class BertLanguageModel(torch.nn.Module):
    def __init__(self, primer, **kwargs):
        super(BertLanguageModel, self).__init__()
        from transformers import BertTokenizer, BertModel
        self.name = 'BERT'
        self.primer = primer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.out_dim = 1024

    def forward(self, text, device, skip_primer=False):
        if skip_primer:
            primed_tokens = text
        else:
            primed_tokens = [self.primer.format(x) for x in text]
        primed_tokens = self.tokenizer(primed_tokens,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True).to(device)
        language_embeds = self.model(**primed_tokens).pooler_output
        return language_embeds.type(torch.float32)


#############################################################################
def adjust_text(input_text, maxlen=30):
    text = ''
    count = 0
    for p, c in enumerate(input_text.split(' ')):
        if p:
            text += ' '
        if count > maxlen and len(text) > 0:
            text += '\n'
            count -= maxlen
        text += c
        count += len(c)
    return text


def reembed_dict_in_language(language_model, label_dict, device):
    print('Getting language embeddings...')

    sorted_values = list(label_dict.values())
    unique_labs = {key: None for key in np.unique(sorted_values)}

    reembed_collect = []
    with torch.no_grad():
        language_embeds = language_model(list(unique_labs.keys()), device,
                                         False).cpu()
        unique_labs = {
            key: language_embed
            for key, language_embed in zip(unique_labs.keys(), language_embeds)
        }

    return {key: unique_labs[value] for key, value in label_dict.items()}


def reembed_in_language(language_model, reassigns_topk, device):
    print('Getting language embeddings...')
    unique_labs = {key: None for key in np.unique(reassigns_topk)}
    reembed_collect = []
    _ = language_model.eval()
    with torch.no_grad():
        language_embeds = language_model(list(unique_labs.keys()), device,
                                         False).cpu()
        unique_labs = {
            key: language_embed
            for key, language_embed in zip(unique_labs.keys(), language_embeds)
        }

    def match(inp):
        return [unique_labs[i] for i in inp]

    reembed_collect = list(map(match, reassigns_topk))
    return torch.stack([torch.stack(x) for x in reembed_collect])


def relabel(model,
            dataloader,
            device,
            datapath='',
            full_label=False,
            topk=5,
            overlap=True):
    was_training = model.training
    _ = model.eval()

    crop_size = dataloader.dataset.crop_size
    base_size = dataloader.dataset.base_size
    dataloader.dataset.crop_size = [299, 299]
    dataloader.dataset.base_size = 320
    dataloader.dataset.provide_transforms()

    if overlap:
        assert topk > 1, 'If you want label overlap, please set topk > 1!'

    with open(datapath + 'imagenet_synsets.txt', 'r') as f:
        imagenet_synsets = f.readlines()
    imagenet_classes = [x.strip() for x in imagenet_synsets]
    imagenet_splits = [line.split(' ') for line in imagenet_synsets]
    key_to_classname = {
        spl[0]: ' '.join(spl[1:]).replace('\n', '')
        for spl in imagenet_splits
    }

    with open(datapath + 'imagenet_classes.txt', 'r') as f:
        imagenet_classes = f.readlines()
    abstract_imagenet_classes = [
        x.strip().replace('\n', '') for x in imagenet_classes
    ]
    imagenet_classes = [key_to_classname[x] for x in abstract_imagenet_classes]

    print('\n')
    iterator = tqdm.tqdm(dataloader, 'Getting ImageNet pseudolabels...')
    memory_collect = []
    train_labels = []
    class_embed_collect = {}
    sample_reassign_topk = []

    for i, data_input in enumerate(iterator):
        with torch.no_grad():
            input = data_input[1]['image']
            out = model(input.to(device))
        for idx, label in zip(out, data_input[0].cpu().detach().numpy()):
            if label not in class_embed_collect:
                class_embed_collect[label] = []
            class_embed_collect[label].append(idx.detach().cpu().numpy())
        train_labels.extend(data_input[0].cpu().detach().numpy().tolist())
        sample_reassign_topk.extend(
            np.array(imagenet_classes)[np.argsort(
                out.detach().cpu().numpy(), axis=1)[:,
                                                    -topk:][:, ::-1]].tolist())

    class_collect_topk = {
        key: np.argsort(np.stack(item, axis=0).mean(axis=0))[-topk:][::-1]
        for key, item in class_embed_collect.items()
    }

    label_reassign_topk = [[] for _ in range(topk)]
    for k in range(topk):
        for label in np.unique(train_labels):
            label_reassign_topk[k].append(
                imagenet_classes[class_collect_topk[label][k]])
    if not full_label:
        label_reassign_topk = [[x.split(', ')[0] for x in y]
                               for y in label_reassign_topk]
        sample_reassign_topk = [[x.split(', ')[0] for x in y]
                                for y in sample_reassign_topk]

    if was_training:
        _ = model.train()

    dataloader.dataset.crop_size = crop_size
    dataloader.dataset.base_size = base_size
    dataloader.dataset.provide_transforms()

    return label_reassign_topk, sample_reassign_topk
