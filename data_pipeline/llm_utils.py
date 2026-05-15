# 新增加了对USRA模型的推理支持，用于更高质量的生成Step by Step Rollouts. 同时不影响原始的InternVL使用和推理
# 在 class LanguageModel 中，保留已有 generate_results 不动，新增一个“对多个 prompt 一次性生成、每个 prompt 返回 num_mc 个最终答案”的方法（放在 generate_results 后面）
import logging
import re
from typing import List, Optional, Union
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from PIL import Image

logger = logging.getLogger('main')


class LanguageModel:
    def __init__(
        self,
        model='OpenGVLab/InternVL2_5-8B',
        max_new_tokens=4096,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    ):
        self.model_path = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = 1.05
        self.is_ursa = 'ursa' in str(self.model_path).lower()
        self.solver_completion_tokens = 0  # 新增：累计生成的新token数

        logger.info(f'Loading model {self.model_path}...')


        mm_limit = {'image': 1} if getattr(self, 'is_ursa', False) else {'image': 8}
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt=mm_limit,
        )
        
        # self.model = LLM(
        #     model=self.model_path,
        #     trust_remote_code=True,
        #     tensor_parallel_size=1,
        #     limit_mm_per_prompt={'image': 8},
        # )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.stop_tokens = ['<|im_end|>\n'.strip()]
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens
        ]
        self.special_tokens = self.tokenizer.all_special_tokens
        self.custom_tokens = ['<step>', '</step>', '<answer>', '</answer>']
        self.special_tokens = [
            token for token in self.special_tokens if token not in self.custom_tokens
        ]
        self.pattern1 = r'|'.join(map(re.escape, self.special_tokens))
        self.pattern2 = r'<step>(.*?)</step>|<answer>(.*?)</answer>'
        logger.info('Model loaded successfully.')

    def generate_results(self, prompt, image_path: Optional[Union[str, List[str]]] = None, num_copies=16):
        inputs = []

        if self.is_ursa:
            user_text = prompt.replace('<image>', '').strip()
            template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image|>{}<|im_end|>
<|im_start|>assistant
"""
            # URSA：仅支持单图，若传入列表则取第一张
            img_obj = None
            if isinstance(image_path, list) and len(image_path) > 0:
                image_path = image_path[0]
            if image_path:
                img_obj = Image.open(image_path).convert('RGB')
            for _ in range(num_copies):
                if img_obj is not None:
                    inputs.append({'prompt': template.format(user_text), 'multi_modal_data': {'image': img_obj}})
                else:
                    inputs.append({'prompt': template.format(user_text)})
            prompt_text_for_log = user_text
        else:
            # InternVL：根据图片数量补足 <image> 占位符；多图时传入列表
            n_images = 0
            imgs = None
            if isinstance(image_path, list):
                n_images = len(image_path)
                if n_images > 0:
                    imgs = [Image.open(p).convert('RGB') for p in image_path]
            elif image_path:
                n_images = 1
                imgs = Image.open(image_path).convert('RGB')

            prompt_with_images = prompt
            if n_images > 0:
                existing = prompt_with_images.count('<image>')
                need = max(0, n_images - existing)
                if need > 0:
                    prompt_with_images = (' '.join(['<image>'] * need) + '\n') + prompt_with_images
            elif '<image>' not in prompt_with_images:
                prompt_with_images = '<image>\n' + prompt_with_images

            messages = [
                {
                    'role': 'system',
                    'content': '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
                },
                {'role': 'user', 'content': prompt_with_images},
            ]
            prompt_text_for_log = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            for _ in range(num_copies):
                if imgs is not None:
                    # 多图：传列表；单图：传单个对象
                    if isinstance(imgs, list):
                        inputs.append({'prompt': prompt_text_for_log, 'multi_modal_data': {'image': imgs}})
                    else:
                        inputs.append({'prompt': prompt_text_for_log, 'multi_modal_data': {'image': imgs}})
                else:
                    inputs.append({'prompt': prompt_text_for_log})

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
        )
        model_outputs = self.model.generate(inputs, sampling_params=sampling_params)
        batch_results = []
        for model_output in model_outputs:
            # 新增：累计该样本的生成token数
            try:
                self.solver_completion_tokens += len(model_output.outputs[0].token_ids)
            except Exception:
                pass

            response = re.sub(self.pattern1, '', model_output.outputs[0].text)

            if self.is_ursa:
                step_pattern = r'(?:^|\n)\s*Step\s*\d+\s*:\s*(.*?)(?=(?:\n\s*Step\s*\d+\s*:)|\n\s*†?\s*Answer\s*:|$)'
                ans_pattern = r'†?\s*Answer\s*:\s*(.*)'
                steps = [s.strip() for s in re.findall(step_pattern, response, flags=re.IGNORECASE|re.DOTALL) if s.strip()]
                ans_matches = re.findall(ans_pattern, response, flags=re.IGNORECASE|re.DOTALL)
                final = ans_matches[-1].strip() if ans_matches else ""
                res = steps + ([final] if final else ([steps[-1]] if steps else []))
            else:
                matches = re.findall(self.pattern2, response, re.DOTALL)
                res = (
                    [match[0] if match[0] else match[1] for match in matches]
                    if matches
                    else []
                )
                res = list(map(str.strip, res))
                if not res:
                    m = re.search(r'Answer:\s*(?:The final answer is\s*)?(.*)', response, re.IGNORECASE)
                    if m and m.group(1).strip():
                        res = [m.group(1).strip()]
                # Fallback 2: 仍未解析到任何片段时，用整段响应作为一个“大 step”
                if not res and response.strip():
                    res = [response.strip()]
                # matches = re.findall(self.pattern2, response, re.DOTALL)
                # step_texts = [m[0].strip() for m in matches if m[0] and m[0].strip()]
                # ans_texts  = [m[1].strip() for m in matches if m[1] and m[1].strip()]

                # if step_texts:
                #     # 正常：有 <step>，返回 steps + 最后一个 <answer>（如果有）
                #     res = step_texts + ([ans_texts[-1]] if ans_texts else [])
                # else:
                #     # 无任何 <step>：用“整段响应”作为一个大 step；同时尽量单独提取一个最终答案
                #     full = response.strip()
                #     ans_final = ans_texts[-1] if ans_texts else ""
                #     if not ans_final:
                #         m = re.search(r'Answer:\s*(?:The final answer is\s*)?(.*)', response, re.IGNORECASE)
                #         if m and m.group(1) and m.group(1).strip():
                #             ans_final = m.group(1).strip()
                #     if full:
                #         res = [full] + ([ans_final] if ans_final else [])
                #     elif ans_final:
                #         res = [ans_final]
                #     else:
                #         res = []

            batch_results.append(res)

        for result in batch_results:
            logger.debug(f'Prompt: {prompt_text_for_log}\nGenerated rollout: {result}')

        return batch_results

    def generate_batch(self, prompts: List[str], image_path: Optional[str] = None, num_mc: int = 16) -> List[List[str]]:
        inputs = []
        # ursa 与 internvl 两套模板保持与 generate_results 一致
        if getattr(self, 'is_ursa', False):
            template = """<|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|image|>{}<|im_end|>
    <|im_start|>assistant
    """
            img_obj = None
            if image_path:
                #from PIL import Image
                img_obj = Image.open(image_path).convert('RGB')
            for p in prompts:
                user_text = p.replace('<image>', '').strip()
                if img_obj:
                    inputs.append({'prompt': template.format(user_text), 'multi_modal_data': {'image': img_obj}})
                else:
                    inputs.append({'prompt': template.format(user_text)})
            get_segments = self._parse_ursa_segments
        else:
            #from vllm.multimodal.utils import fetch_image
            def to_chat_prompt(p: str) -> str:
                if '<image>' not in p:
                    p = '<image>\n' + p
                messages = [
                    {'role': 'system', 'content': '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
                    {'role': 'user', 'content': p},
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            img_obj = None
            if image_path:
                #img_obj = fetch_image('file://' + image_path, allowed_local_media_path='/')
                img_obj = Image.open(image_path).convert('RGB')

            for p in prompts:
                prompt_text = to_chat_prompt(p)
                if img_obj:
                    inputs.append({'prompt': prompt_text, 'multi_modal_data': {'image': img_obj}})
                else:
                    inputs.append({'prompt': prompt_text})
            get_segments = self._parse_internvl_segments

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
            n=max(1, int(num_mc)),  # 关键：每个 prompt 直接生成 num_mc 个采样
        )
        model_outputs = self.model.generate(inputs, sampling_params=sampling_params)

        results: List[List[str]] = []
        for mo in model_outputs:
            answers_this_prompt: List[str] = []
            for out in mo.outputs:
                # 新增：累计该采样的生成token数
                try:
                    self.solver_completion_tokens += len(out.token_ids)
                except Exception:
                    pass

                response = re.sub(self.pattern1, '', out.text)
                segments = get_segments(response)
                # 仅保留“最终答案”（与 mc_q 使用一致）
                final_ans = segments[-1].strip() if segments else ""
                if not final_ans:
                    m = re.search(r'Answer:\s*(?:The final answer is\s*)?(.*)', response, re.IGNORECASE)
                    if m and m.group(1).strip():
                        final_ans = m.group(1).strip()
                answers_this_prompt.append(final_ans)
            results.append(answers_this_prompt)
        return results


    def generate_batch_segments(self, prompts: List[str], image_path: Optional[Union[str, List[str]]] = None, num_mc: int = 16) -> List[List[List[str]]]:
        inputs = []
        # ursa 与 internvl 两套模板保持与 generate_results / generate_batch 一致
        if getattr(self, 'is_ursa', False):
            template = """<|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|image|>{}<|im_end|>
    <|im_start|>assistant
    """
            # URSA：仅支持单图，若传入列表则取第一张
            img_obj = None
            if isinstance(image_path, list) and len(image_path) > 0:
                image_path = image_path[0]
            if image_path:
                img_obj = Image.open(image_path).convert('RGB')
            for p in prompts:
                user_text = p.replace('<image>', '').strip()
                if img_obj is not None:
                    inputs.append({'prompt': template.format(user_text), 'multi_modal_data': {'image': img_obj}})
                else:
                    inputs.append({'prompt': template.format(user_text)})
            get_segments = self._parse_ursa_segments
        else:
            # InternVL：按图片数补足 <image>，多图传列表
            n_images = 0
            imgs = None
            if isinstance(image_path, list):
                n_images = len(image_path)
                if n_images > 0:
                    imgs = [Image.open(p).convert('RGB') for p in image_path]
            elif image_path:
                n_images = 1
                imgs = Image.open(image_path).convert('RGB')

            def to_chat_prompt(p: str) -> str:
                pp = p
                if n_images > 0:
                    existing = pp.count('<image>')
                    need = max(0, n_images - existing)
                    if need > 0:
                        pp = (' '.join(['<image>'] * need) + '\n') + pp
                elif '<image>' not in pp:
                    pp = '<image>\n' + pp
                messages = [
                    {'role': 'system', 'content': '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
                    {'role': 'user', 'content': pp},
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            for p in prompts:
                prompt_text = to_chat_prompt(p)
                if imgs is not None:
                    if isinstance(imgs, list):
                        inputs.append({'prompt': prompt_text, 'multi_modal_data': {'image': imgs}})
                    else:
                        inputs.append({'prompt': prompt_text, 'multi_modal_data': {'image': imgs}})
                else:
                    inputs.append({'prompt': prompt_text})
            get_segments = self._parse_internvl_segments

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
            n=max(1, int(num_mc)),  # 每个 prompt 直接生成 num_mc 个采样
        )
        model_outputs = self.model.generate(inputs, sampling_params=sampling_params)

        results: List[List[List[str]]] = []
        for mo in model_outputs:
            samples_one_prompt: List[List[str]] = []
            for out in mo.outputs:
                try:
                    self.solver_completion_tokens += len(out.token_ids)
                except Exception:
                    pass

                response = re.sub(self.pattern1, '', out.text)
                segments = get_segments(response)
                # 兜底：若解析不到结构，但有文本，至少保留整段响应作为一个片段
                if not segments and response.strip():
                    segments = [response.strip()]
                samples_one_prompt.append([s.strip() for s in segments if isinstance(s, str)])
            results.append(samples_one_prompt)
        return results

    def generate_batch_texts(self, prompts: List[str], image_path: Optional[Union[str, List[str]]] = None, num_mc: int = 16) -> List[List[str]]:
        inputs = []
        # 与 generate_batch / generate_batch_segments 相同的模板与多模态构造，但不做解析，直接返回原始文本
        if getattr(self, 'is_ursa', False):
            template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image|>{}<|im_end|>
<|im_start|>assistant
"""
            # URSA：仅支持单图，若传入列表则取第一张
            img_obj = None
            if isinstance(image_path, list) and len(image_path) > 0:
                image_path = image_path[0]
            if image_path:
                img_obj = Image.open(image_path).convert('RGB')
            for p in prompts:
                user_text = p.replace('<image>', '').strip()
                if img_obj is not None:
                    inputs.append({'prompt': template.format(user_text), 'multi_modal_data': {'image': img_obj}})
                else:
                    inputs.append({'prompt': template.format(user_text)})
        else:
            n_images = 0
            imgs = None
            if isinstance(image_path, list):
                n_images = len(image_path)
                if n_images > 0:
                    imgs = [Image.open(p).convert('RGB') for p in image_path]
            elif image_path:
                n_images = 1
                imgs = Image.open(image_path).convert('RGB')

            def to_chat_prompt(p: str) -> str:
                pp = p
                if n_images > 0:
                    existing = pp.count('<image>')
                    need = max(0, n_images - existing)
                    if need > 0:
                        pp = (' '.join(['<image>'] * need) + '\n') + pp
                elif '<image>' not in pp:
                    pp = '<image>\n' + pp
                messages = [
                    {'role': 'system', 'content': '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
                    {'role': 'user', 'content': pp},
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            for p in prompts:
                prompt_text = to_chat_prompt(p)
                if imgs is not None:
                    if isinstance(imgs, list):
                        inputs.append({'prompt': prompt_text, 'multi_modal_data': {'image': imgs}})
                    else:
                        inputs.append({'prompt': prompt_text, 'multi_modal_data': {'image': imgs}})
                else:
                    inputs.append({'prompt': prompt_text})

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
            n=max(1, int(num_mc)),
        )
        model_outputs = self.model.generate(inputs, sampling_params=sampling_params)

        results: List[List[str]] = []
        for mo in model_outputs:
            texts_one_prompt: List[str] = []
            for out in mo.outputs:
                try:
                    self.solver_completion_tokens += len(out.token_ids)
                except Exception:
                    pass
                response = re.sub(self.pattern1, '', out.text)
                texts_one_prompt.append(response.strip())
            results.append(texts_one_prompt)
        return results

    def _parse_internvl_segments(self, response: str) -> List[str]:
        matches = re.findall(self.pattern2, response, re.DOTALL)
        res = [m[0] if m[0] else m[1] for m in matches] if matches else []
        return list(map(str.strip, res))

    def _parse_ursa_segments(self, response: str) -> List[str]:
        step_pattern = r'(?:^|\n)\s*Step\s*\d+\s*:\s*(.*?)(?=(?:\n\s*Step\s*\d+\s*:)|\n\s*†?\s*Answer\s*:|$)'
        ans_pattern = r'†?\s*Answer\s*:\s*(.*)'
        steps = [s.strip() for s in re.findall(step_pattern, response, flags=re.IGNORECASE|re.DOTALL) if s.strip()]
        ans_matches = re.findall(ans_pattern, response, flags=re.IGNORECASE|re.DOTALL)
        final = ans_matches[-1].strip() if ans_matches else ""
        return steps + ([final] if final else ([steps[-1]] if steps else []))
    
    def get_solver_completion_tokens(self) -> int:
        return int(getattr(self, 'solver_completion_tokens', 0))