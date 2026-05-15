import argparse
import itertools
import json
import math
import os
import random
import subprocess
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from internvl.conversation import get_conv_template
from internvl.model import split_model
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat.modeling_internvl_chat_beta_binom import \
    InternVLChatModel as InternVLChatBetaBinomModel
from internvl.train.dataset import build_transform, dynamic_preprocess

ds_collections = {
    'mathvision_prm': {
        'root': 'datasets/MathVision/extracted_images',
        'annotation': 'datasets/MathVision/MathVision_rollout_annotation_InternVL8B_oversample.json',
    }
}


def load_model_and_tokenizer_beta(args):
    use_auto_map = bool(args.auto and torch.cuda.device_count() > 1)
    if use_auto_map:
        config = InternVLChatConfig.from_pretrained(args.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
        # Ensure the beta-binomial head is assigned in auto device map.
        if 'kappa_head' not in device_map:
            device_map['kappa_head'] = 0
    kwargs = {'device_map': device_map} if use_auto_map else {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, use_fast=False
    )
    model = InternVLChatBetaBinomModel.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        **kwargs,
    ).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not use_auto_map:
        model = model.cuda()
    return model, tokenizer


@torch.no_grad()
def batch_prm_mu_kappa(
    model,
    tokenizer,
    pixel_values,
    questions,
    num_patches_list,
    verbose=False,
):
    prm_token_id = tokenizer.convert_tokens_to_ids('<prm>')
    reward_token_ids = tokenizer.convert_tokens_to_ids(['Yes', 'No'])
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    queries = []
    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        template = get_conv_template(model.template)
        template.append_message(template.roles[0], '')
        template.append_message(template.roles[1], question)
        query = template.get_prompt()

        image_tokens = '<img>' + '<IMG_CONTEXT>' * model.num_image_token * num_patches + '</img>'
        query = query.replace('<image>', image_tokens, 1)
        queries.append(query)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
    tokenizer.padding_side = old_padding_side

    device = pixel_values.device
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)

    outputs = model.forward(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long, device=device),
        output_hidden_states=True,
        return_dict=True,
    )
    logits = outputs.logits
    placeholder_mask = input_ids == prm_token_id
    selected_logits = logits[placeholder_mask]
    selected_logits = selected_logits[..., reward_token_ids]
    if selected_logits.numel() == 0:
        empty = selected_logits.new_zeros((0,), dtype=torch.float32)
        return empty, empty

    # Keep mu from Yes/No relative evidence.
    mu = torch.softmax(selected_logits, dim=-1)[..., 0]
    if hasattr(model, 'kappa_head') and model.kappa_head is not None:
        hidden = outputs.hidden_states[-1]
        prm_h = hidden[placeholder_mask]
        z_kappa = model.kappa_head(prm_h).squeeze(-1)
        kappa_min = float(getattr(model, 'beta_binom_kappa_min', 1e-3))
        kappa = torch.nn.functional.softplus(z_kappa) + kappa_min
    else:
        # Backward compatibility for old checkpoints.
        kappa_floor = float(getattr(model, 'beta_binom_kappa_floor', 2.0))
        kappa = torch.nn.functional.softplus(selected_logits[..., 1]) + kappa_floor
    return mu, kappa

def _pick_question_text(data_item):
    """
    Be robust across different annotation schemas.
    Priority:
      - query (MathVista parquet-style)
      - query_cot (MathVerse-style)
      - question (seed/rollout builders often store this)
    """
    q = (
        data_item.get('query')
        or data_item.get('query_cot')
        or data_item.get('question')
        or ''
    )
    return '' if q is None else str(q)


def _pick_image_path(root, data_item):
    """
    Be robust across different annotation schemas.
    Priority:
      - any existing absolute image_path
      - image joined with root
      - basename/image_path fallbacks under root
    """
    candidates = []

    raw_abs = data_item.get('image_path')
    raw_rel = data_item.get('image')

    if raw_abs is not None:
        raw_abs = str(raw_abs)
        candidates.append(raw_abs)

    if raw_rel is not None:
        raw_rel = str(raw_rel)
        candidates.append(os.path.join(root, raw_rel))
        candidates.append(os.path.join(root, os.path.basename(raw_rel)))

    if raw_abs is not None:
        candidates.append(os.path.join(root, raw_abs))
        candidates.append(os.path.join(root, os.path.basename(raw_abs)))

    seen = set()
    normalized = []
    for path in candidates:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        normalized.append(norm_path)

    for path in normalized:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f'No valid image path found for item id={data_item.get("id")}. '
        f'Tried: {normalized}'
    )


def collate_fn(batches):
    pixel_values = batches[0]['pixel_values']
    prompts = batches[0]['prompts']
    steps_lens = batches[0]['steps_lens']
    data_items = batches[0]['data_item']
    return pixel_values, prompts, steps_lens, data_items


class MathVistaPRMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        annotation,
        input_size=224,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        self.root = root
        self.data = json.load(open(annotation))
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        image = _pick_image_path(self.root, data_item)

        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        question = _pick_question_text(data_item)

        prompts = []
        steps_lens = []
        if 'solutions_splits' not in data_item:
            raise KeyError(
                "Missing required key 'solutions_splits' in annotation item. "
                "This evaluator expects rollout annotations produced by the build scripts."
            )
        for solution_split in data_item['solutions_splits']:
            solution = '<prm>'.join(solution_split) + '<prm>'
            prompt = f'Question: {question}\nProcess: {solution}'
            prompts.append(prompt)
            steps_lens.append(len(solution_split))

        return {
            'pixel_values': pixel_values,
            'prompts': prompts,
            'steps_lens': steps_lens,
            'data_item': data_item,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MathVistaPRMDataset(
            root=ds_collections[ds_name]['root'],
            annotation=ds_collections[ds_name]['annotation'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for idx, (pixel_values, prompts, steps_lens, data_item) in tqdm(
            enumerate(dataloader)
        ):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            prm_scores_flattened = []
            prm_mu_flattened = []
            prm_kappa_flattened = []
            prm_sigma_flattened = []
            for i in range(0, len(prompts), args.mini_batch_size):
                curr_bs = min(args.mini_batch_size, len(prompts) - i)
                curr_pixel_values = torch.cat([pixel_values] * curr_bs, dim=0)
                curr_questions = prompts[i : i + curr_bs]
                curr_num_patches = [pixel_values.shape[0]] * curr_bs
                mu, kappa = batch_prm_mu_kappa(
                    model=model,
                    tokenizer=tokenizer,
                    pixel_values=curr_pixel_values,
                    questions=curr_questions,
                    num_patches_list=curr_num_patches,
                    verbose=False,
                )
                sigma = torch.sqrt(mu * (1.0 - mu) / (kappa + 1.0))
                if args.score_mode == 'mu_minus_lambda_sigma':
                    score = mu - args.risk_lambda * sigma
                else:
                    score = mu

                prm_scores_flattened.extend(score.tolist())
                prm_mu_flattened.extend(mu.tolist())
                prm_kappa_flattened.extend(kappa.tolist())
                prm_sigma_flattened.extend(sigma.tolist())

            data_item['prm_scores'] = []
            data_item['prm_mu'] = []
            data_item['prm_kappa'] = []
            data_item['prm_sigma'] = []
            curr_len = 0
            for i in range(len(steps_lens)):
                data_item['prm_scores'].append(
                    prm_scores_flattened[curr_len : curr_len + steps_lens[i]]
                )
                data_item['prm_mu'].append(
                    prm_mu_flattened[curr_len : curr_len + steps_lens[i]]
                )
                data_item['prm_kappa'].append(
                    prm_kappa_flattened[curr_len : curr_len + steps_lens[i]]
                )
                data_item['prm_sigma'].append(
                    prm_sigma_flattened[curr_len : curr_len + steps_lens[i]]
                )
                curr_len += steps_lens[i]

            for i in range(len(data_item['prm_scores'])):
                assert len(data_item['prm_scores'][i]) == steps_lens[i]

            print(f'Pred: {data_item["prm_scores"]}')
            outputs.append(data_item)

            if idx % 50 == 0:
                torch.distributed.barrier()

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            output_path = os.path.join(args.out_dir, results_file)
            json.dump(
                merged_outputs, open(output_path, 'w'), indent=4, ensure_ascii=False
            )
            print('Results saved to {}'.format(output_path))

            cmd = f'python eval/prm/extract_calculate.py --output_file {results_file} --output_dir {args.out_dir}'
            print(cmd)
            os.system(cmd)

            if not args.skip_uncertainty_diagnose:
                input_json = output_path
                out_json = output_path.replace('.json', '_uncertainty_diag.json')
                diag_cmd = [
                    'python',
                    'eval/prm/diagnose_mathvista_uncertainty.py',
                    '--input-json',
                    input_json,
                    '--out-json',
                    out_json,
                    '--lambdas',
                    args.diag_lambdas,
                    '--topq-grid',
                    args.diag_topq_grid,
                    '--budget-q-grid',
                    args.diag_budget_q_grid,
                    '--soft-q-grid',
                    args.diag_soft_q_grid,
                    '--soft-temp-grid',
                    args.diag_soft_temp_grid,
                    '--quantile-grid',
                    args.diag_quantile_grid,
                    '--softmin-temp-grid',
                    args.diag_softmin_temp_grid,
                    '--mink-frac-grid',
                    args.diag_mink_frac_grid,
                ]
                print(' '.join(diag_cmd))
                subprocess.run(diag_cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--root', type=str, default='', help='Override image root directory for all datasets')
    parser.add_argument('--annotation', type=str, default='', help='Override annotation JSON path for all datasets')
    parser.add_argument('--mini-batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument(
        '--score-mode',
        type=str,
        default='mu',
        choices=['mu', 'mu_minus_lambda_sigma'],
        help='Scoring mode for BoN ranking.',
    )
    parser.add_argument(
        '--risk-lambda',
        type=float,
        default=0.5,
        help='Risk penalty lambda for mu-lambda*sigma mode.',
    )
    parser.add_argument(
        '--skip-uncertainty-diagnose',
        action='store_true',
        help='Skip post-eval uncertainty diagnosis sweep.',
    )
    parser.add_argument(
        '--diag-lambdas',
        type=str,
        default='0,0.1,0.2,0.35,0.5,0.7,1.0,1.5',
        help='Comma-separated lambdas for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-topq-grid',
        type=str,
        default='0.7,0.8,0.9',
        help='Comma-separated top-q grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-budget-q-grid',
        type=str,
        default='0.7,0.8,0.9',
        help='Comma-separated budget-q grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-soft-q-grid',
        type=str,
        default='0.7,0.8,0.9',
        help='Comma-separated soft-q grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-soft-temp-grid',
        type=str,
        default='0.01,0.02,0.05,0.1',
        help='Comma-separated soft temperature grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-quantile-grid',
        type=str,
        default='0.2,0.3,0.4,0.5',
        help='Comma-separated quantile grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-softmin-temp-grid',
        type=str,
        default='5,10,20',
        help='Comma-separated softmin temperature grid for uncertainty diagnosis.',
    )
    parser.add_argument(
        '--diag-mink-frac-grid',
        type=str,
        default='0.1,0.2,0.3',
        help='Comma-separated min-k fraction grid for uncertainty diagnosis.',
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    # Optional overrides to avoid editing source for different annotation files.
    if args.root:
        for _ds in args.datasets:
            ds_collections[_ds]['root'] = args.root
    if args.annotation:
        for _ds in args.datasets:
            ds_collections[_ds]['annotation'] = args.annotation

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer_beta(args)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
