# 新增加了对USRA模型的推理支持，用于更高质量的生成Step by Step Rollouts. 同时不影响原始的InternVL使用和推理
# 增加断点续传功能
import argparse
import json
import os
import re
import logging
import requests
import time
from tqdm import tqdm

IMAGE_ROOT = ""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('build_eval_annotation')


def atomic_write_json_array(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def select_best_rollouts(seg_lists, k):
    # 去重（完全相同的段落序列）
    seen = set()
    deduped = []
    for segs in seg_lists:
        key = tuple(segs)
        if key not in seen:
            seen.add(key)
            deduped.append(segs)

    # 先选“像CoT”的：长度>=2（至少一个 <step> + 最后答案），且都非空
    valid = [s for s in deduped if len(s) >= 2 and all(x.strip() for x in s)]
    # 排序：先按步数多优先，其次按总字数多优先（更完整）
    valid.sort(key=lambda s: (len(s), sum(len(x) for x in s)), reverse=True)

    result = valid[:k]
    if len(result) < k:
        # 兜底：允许只有答案的（len>=1），依然按相同排序补齐
        rest = [s for s in deduped if len(s) >= 1 and all(x.strip() for x in s) and s not in result]
        rest.sort(key=lambda s: (len(s), sum(len(x) for x in s)), reverse=True)
        result.extend(rest[:k - len(result)])
    return result


def _format_steps_xml(steps):
    if not steps:
        return ""
    if len(steps) >= 2:
        body = "".join([f"<step>{s}</step>" for s in steps[:-1]]) + f"<answer>{steps[-1]}</answer>"
    else:
        body = steps[0]
    return body


def judge_quality_score(api_base, model_name, question, steps, timeout=30, retries=3, sleep_between=0.0):
    """
    让部署的 LLM 对“推理链质量”打分，返回 0~10 浮点分；不评判最终答案对错。
    """
    url = api_base.rstrip("/") + "/chat/completions"
    content = f"""You are evaluating the reasoning quality of a chain-of-thought for a math/logic question.
Score ONLY the reasoning quality (clarity, coherence, step coverage, on-topic, non-hallucination, usefulness). 
Ignore whether the final answer is correct or not.
Return ONLY a single number between 0 and 10 (decimals allowed). No extra text.

Question:
{question}
--------------------------------
Candidate reasoning (XML):
{_format_steps_xml(steps)}
--------------------------------
"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": content}]}],
        "max_tokens": 8,
        "temperature": 0.0,
    }

    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            resp = data["choices"][0]["message"]["content"]
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(resp))
            if m:
                val = float(m.group(1))
                return max(0.0, min(10.0, val))
        except Exception as e:
            last_err = e
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue
    logger.warning(f"quality judge failed, fallback score=0; err={last_err}")
    return 0.0


def select_by_llm_quality(seg_lists, k, api_base, model_name, question, timeout, retries):
    # 去重与基础过滤：去掉完全空的
    seen = set()
    cand = []
    for segs in seg_lists:
        if not segs or not any(s.strip() for s in segs):
            continue
        key = tuple(segs)
        if key in seen:
            continue
        seen.add(key)
        cand.append(segs)

    # 打分：不以对错为标准，只以推理质量评分
    scored = []
    for segs in cand:
        score = judge_quality_score(
            api_base=api_base,
            model_name=model_name,
            question=question,
            steps=segs,
            timeout=timeout,
            retries=max(1, retries),
            sleep_between=0.0
        )
        scored.append((score, segs))

    # 排序：先按质量分，其次用轻量 tie-break（步数与总字数）但不支配
    scored.sort(key=lambda t: (t[0], len(t[1]), sum(len(x) for x in t[1])), reverse=True)
    chosen = [segs for _, segs in scored[:k]]

    # 兜底：若仍不足K（极少见），用剩余非空补齐
    if len(chosen) < k:
        rest = [segs for _, segs in scored if segs not in chosen]
        chosen.extend(rest[:k - len(chosen)])
    return chosen[:k]

def make_cot_prompt(question):
    return f"""<image>
You are an AI assistant that must strictly follow the response protocol below.

1) Reasoning phase:
- Analyze the question thoroughly and consider plausible solution strategies.
- Proceed step by step and wrap each reasoning step in <step>...</step>.
- Keep each step concise, factual, and focused on the question and the image (if provided).
- Do NOT include any text outside <step> tags in this phase.

2) Final answer:
- After the reasoning steps, add a single blank line.
- Provide the final answer wrapped in exactly one <answer>...</answer> tag.
- The final answer must be self-contained and must NOT reference the reasoning text.
- Output nothing outside the required tags and nothing after </answer>.

Question: {question}"""

# def make_cot_prompt(question):
#     return f"""<image>
#             Please solve the problem and output strictly in XML tags only:
#             - Wrap each reasoning step in <step>...</step>
#             - Put the final answer in a single <answer>...</answer> tag
#             Question: {question}"""

#     return f"""<image>
# Solve the problem. Output STRICTLY and ONLY using these XML tags:
# - Wrap each reasoning step in <step>...</step>.
# - Put the final answer in ONE <answer>...</answer> tag as the LAST tag.
# Do NOT output anything outside these tags. No extra text after </answer>.
# Question: {question}"""

def make_cot_prompt_ursa(question):
    # URSA 风格的提示词（不包含 XML 约束；URSA 模板内会注入 <|image|>）
    return f"""you are given a math problem image, please solve the problem step by step. Question:{question}"""


def extract_steps_and_final(segments):
    if not segments:
        return [], ""
    if len(segments) == 1:
        return [], segments[0].strip()
    steps = [s.strip() for s in segments[:-1]]
    final = segments[-1].strip()
    return steps, final


def judge_yes_no(api_base, model_name, question, correct_answer, model_answer, timeout=30, retries=5):
    """
    仅依赖 judge 大模型的 Yes/No 输出：
    - 最多重试 retries 次
    - 'yes'/'no' 使用包含匹配，容忍 "Yes.", "yes," 等
    - 若所有重试都未命中，返回 0 (No)
    """
    url = api_base.rstrip("/") + "/chat/completions"
    payload_template = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are given a question, the correct answer and a model's answer. Please determine if the model's answer matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer.
Only the correctness of the model's answer matters.
Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{correct_answer}
--------------------------------
Model's Answer:
{model_answer}
--------------------------------"""
                    }
                ]
            }
        ],
        "max_tokens": 8,
        "temperature": 0.0
    }

    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload_template, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            content_str = content if isinstance(content, str) else str(content)
            lc = content_str.lower().strip()
            logger.debug(f'【Judge Model Raw Output #{attempt+1}】: "{content_str}"')

            if 'yes' in lc:
                return 1
            if 'no' in lc:
                return 0
            # 否则继续重试
        except Exception as e:
            logger.warning(f'Judge request failed at attempt {attempt+1}/{retries}: {e}')
    # 所有重试失败或无明确 yes/no 时，默认判为 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始数据 JSON：列表，每项至少含 question, correct_answer, image_path(可选)")
    parser.add_argument("--output", required=True, help="输出 annotation JSON（评测标注文件）")
    parser.add_argument("--generator_model", default="OpenGVLab/InternVL2_5-8B", help="用于生成候选推理链的本地模型")
    parser.add_argument("--num_rollouts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)

    parser.add_argument("--judge_api_base", required=True, help="判别用 vLLM OpenAI 服务器，如 http://127.0.0.1:8000/v1")
    parser.add_argument("--judge_model", required=True, help="判别用模型名（served-model-name）")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=5)

    parser.add_argument("--oversample", type=float, default=2.0, help="超采样因子，用于先多生成再筛选为 num_rollouts")
    parser.add_argument("--select_by_llm_quality", action="store_true", help="使用部署的LLM按推理质量打分来选出前K")

    parser.add_argument("--flush_every", type=int, default=1, help="每处理多少条样本将累计结果原子写入输出，支持断点续跑。")
    parser.add_argument(
        "--image_root",
        default="",
        help="Root directory used to resolve relative image_path/image fields. Defaults to the input file directory.",
    )

    args = parser.parse_args()

    raw = json.load(open(args.input, "r", encoding="utf-8"))
    total = len(raw)
    logger.info(f"Total samples: {total}")

    input_dir = os.path.dirname(os.path.abspath(args.input))
    image_root = args.image_root or input_dir or IMAGE_ROOT

    # 续跑：如已有部分结果，基于其长度续算
    out_items = []
    done_n = 0
    if os.path.exists(args.output):
        try:
            existing = json.load(open(args.output, "r", encoding="utf-8"))
            if isinstance(existing, list):
                out_items = existing[:]
                done_n = len(out_items)
                logger.info(f"Resume detected: found {done_n} items in {args.output}, remaining {total - done_n}.")
            else:
                logger.warning(f"Existing output is not a list, ignoring: {args.output}")
        except Exception as e:
            logger.warning(f"Failed to load existing {args.output}, ignore resume. err={e}")

    if done_n >= total:
        atomic_write_json_array(args.output, out_items)
        print(f"Saved annotation to {args.output}")
        return

    # 自动识别是否使用 URSA 模板（不会影响 InternVL）
    is_ursa = 'ursa' in str(args.generator_model).lower()

    from data_pipeline.llm_utils import LanguageModel

    # 本地生成器：直接使用 vLLM Python（不走 HTTP）
    gen_lm = LanguageModel(
        model=args.generator_model,
        max_new_tokens=2048,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    flush_every = max(1, int(args.flush_every))

    for idx in tqdm(range(done_n, total), desc="Building annotation", initial=done_n, total=total):
        ex = raw[idx]
        q = ex["question"]
        ans = ex["correct_answer"]

        raw_img = ex.get("image_path") or ex.get("image")
        if raw_img:
            image_path = raw_img if os.path.isabs(raw_img) else os.path.normpath(os.path.join(image_root, raw_img))
        else:
            image_path = ""
            print("empty image_path")

        prompt = make_cot_prompt_ursa(q) if is_ursa else make_cot_prompt(q)

        # 超采样
        first_n = max(args.num_rollouts, int(args.num_rollouts * args.oversample))
        seg_lists_all = gen_lm.generate_results(
            prompt,
            image_path=image_path,
            num_copies=first_n
        )

        if args.select_by_llm_quality:
            selected = select_by_llm_quality(
                seg_lists_all,
                args.num_rollouts,
                api_base=args.judge_api_base,
                model_name=args.judge_model,
                question=q,
                timeout=args.timeout,
                retries=args.retries
            )
            if len(selected) < args.num_rollouts:
                need = args.num_rollouts - len(selected)
                more_n = max(need, int(need * args.oversample))
                seg_lists_more = gen_lm.generate_results(
                    prompt,
                    image_path=image_path,
                    num_copies=more_n
                )
                seg_lists_all.extend(seg_lists_more)
                selected = select_by_llm_quality(
                    seg_lists_all,
                    args.num_rollouts,
                    api_base=args.judge_api_base,
                    model_name=args.judge_model,
                    question=q,
                    timeout=args.timeout,
                    retries=args.retries
                )
            seg_lists = selected
        else:
            seg_lists = select_best_rollouts(seg_lists_all, args.num_rollouts)

        solutions_splits = []
        labels = []
        for segs in seg_lists:
            steps, final_answer = extract_steps_and_final(segs)
            if not final_answer:
                final_answer = steps[-1] if steps else ""
            solutions_splits.append(steps + ([final_answer] if (final_answer and (not steps or steps[-1].strip() != final_answer.strip())) else []))
            lab = judge_yes_no(
                api_base=args.judge_api_base,
                model_name=args.judge_model,
                question=q,
                correct_answer=ans,
                model_answer=final_answer,
                timeout=args.timeout,
                retries=args.retries
            )
            labels.append(lab)

        item = {
            "image_path": image_path or "",
            "image": os.path.basename(image_path) if image_path else "",
            "question": q,
            "query_cot": q,
            "solutions_splits": solutions_splits,
            "labels": labels
        }
        if "id" in ex:
            item["id"] = ex["id"]

        out_items.append(item)

        # 周期性原子写回，作为断点
        if ((len(out_items) - done_n) % flush_every) == 0:
            atomic_write_json_array(args.output, out_items)

    atomic_write_json_array(args.output, out_items)
    print(f"Saved annotation to {args.output}")

if __name__ == "__main__":
    main()
