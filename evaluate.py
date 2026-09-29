import argparse
import os
import sys
import json

import torch
from transformers import GPTNeoXForCausalLM


# =========================
# CONFIG
# =========================
EVAL_CONFIG = {
    "igsm_root": "./iGSM",
}

TEST_HASH_BIN = list(range(17, 23))

EVAL_DIFFICULTIES_PRESETS = {
    "med": {
        "max_new_tokens": 2048,
        "levels": [
            {"name": "op<=15 (in-dist)", "max_op": 15, "max_edge": 20},
            {"name": "op=15 (in-dist)", "max_op": 15, "max_edge": 20},
            {"name": "op=20 (OOD)", "max_op": 20, "max_edge": 20},
            {"name": "op=21 (OOD)", "max_op": 21, "max_edge": 20},
            {"name": "op=22 (OOD)", "max_op": 22, "max_edge": 20},
            {"name": "op=23 (OOD)", "max_op": 23, "max_edge": 20},
        ],
    },
}


TOKEN_START_PROBLEM = 222
TOKEN_START_SOLUTION = 223
TOKEN_START_ANSWER = 224
TOKEN_EOS = 50256


# =========================
# UTILS
# =========================
def _add_igsm_to_path(igsm_root: str):
    igsm_root = os.path.abspath(igsm_root)
    if igsm_root not in sys.path:
        sys.path.insert(0, igsm_root)


def _load_model(checkpoint_path: str, device: str):
    print(f"Loading model from {checkpoint_path}...")
    model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def _extract_answer(generated_tokens):
    _add_igsm_to_path(EVAL_CONFIG["igsm_root"])
    from tools.tools import tokenizer

    try:
        ans_start = generated_tokens.index(TOKEN_START_ANSWER)
        ans_tokens = generated_tokens[ans_start + 1 :]
        if TOKEN_EOS in ans_tokens:
            ans_tokens = ans_tokens[: ans_tokens.index(TOKEN_EOS)]
        return tokenizer.decode(ans_tokens).strip()
    except ValueError:
        return ""


def _extract_solution(generated_tokens):
    _add_igsm_to_path(EVAL_CONFIG["igsm_root"])
    from tools.tools import tokenizer

    try:
        ans_start = generated_tokens.index(TOKEN_START_ANSWER)
        sol_tokens = generated_tokens[:ans_start]
        return tokenizer.decode(sol_tokens).strip()
    except ValueError:
        return ""


def _check_answer(pred, gt):
    return pred.strip() == gt.strip()


def _parsing_stats(predicted_solution, ground_prob):
    from tools.sol_parser import Parser

    try:
        parsed = Parser(predicted_solution)

        correct_parse = parsed.parsed
        incorrect_refer, _ = parsed.correct_refer()
        incorrect_calcs, _ = parsed.correct_cal()
        excess_definitions, incorrect_order, _ = parsed.correct_order()

        parsed.parse(ground_prob)

        return {
            "correct_parse": correct_parse,
            "incorrect_refer": incorrect_refer,
            "incorrect_calcs": incorrect_calcs,
            "excess_definitions": excess_definitions,
            "incorrect_order": incorrect_order,
            "num_non_appear": len(parsed.non_appear_lst),
            "num_non_nece": len(parsed.non_nece_lst),
            "num_incorrect_elements": len(parsed.incorrect_lst),
        }

    except Exception:
        return {
            "correct_parse": False,
            "incorrect_refer": 0,
            "incorrect_calcs": 0,
            "excess_definitions": 0,
            "incorrect_order": 0,
            "num_non_appear": 0,
            "num_non_nece": 0,
            "num_incorrect_elements": 0,
        }



def problem_structure(p, reask_info=None):
    """Serializable dependency structure of an iGSM Problem: one entry per
    parameter with its name, parent names, value, and whether it is necessary
    for the query / stated in the text. Saved per row so any later check
    (validity, minimality, wrong-parent steps) is a lookup, not a regeneration."""
    nec = set(p.topological_order)
    stated_set = set(x for x in p.problem_order if x[0] == 0)
    graph = getattr(p, "whole_template", None) or p.template
    out = []
    for node in graph.nodes:
        if node == p.rand or (node[0] == 0 and node not in stated_set):
            continue
        def _name(x):
            try:
                return p.get_ntn(param=x)
            except Exception:
                return str(x)
        val = p.lookup[node].a if node in p.lookup else None
        entry = {
            "name": _name(node),
            "parents": [_name(x) for x in graph.predecessors(node) if x != p.rand],
            "value": None if val is None else int(val),
            "necessary": node in nec,
            "stated": node[0] == 0,
        }
        if reask_info:
            entry["in_original_chain"] = node in set(reask_info["orig_chain"])
            entry["downstream_of_query"] = node in reask_info["downstream_of_query"]
        out.append(entry)
    return {"query": _name(p.ques_idx), "params": out}

# =========================
# CORE EVAL
# =========================
def evaluate_difficulty(
    model,
    device,
    max_op,
    max_edge,
    num_problems,
    max_new_tokens,
    seed,
    exact_op=None,
    gen_batch=1,
    layer_widths=None,
    max_prob_tokens=0,
    depth=None,
    reask_queries=False,
    problem_cache=None,
    build_only=False,
):
    _add_igsm_to_path(EVAL_CONFIG["igsm_root"])
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import tokenizer, fix_seed
    from probes.gen_geometry import make_id_gen_class
    GenClass = make_id_gen_class(IdGen, layer_widths, depth)

    fix_seed(seed)

    correct = 0
    results = []

    # Problem generation stays a single sequential pass regardless of
    # gen_batch: the RNG stream defines the problem set, and batching must
    # not change which problems an eval sees.
    #
    # A fresh IdGen per problem: IdGen.__init__ draws the target op once, so a
    # reused generator would give every problem of a ranged level the same op.
    problems = []
    n_rejected_long = 0
    import random as _random
    reask_rng = _random.Random(seed + 1)
    from probes.reask_eval import reask as _reask
    # Problem cache: the evaluation problems of a level depend only on
    # (max_op, max_edge, exact_op, seed, n), so they are generated once and shared by
    # every model. Only for the standard protocol (no geometry / reask / length cap).
    import pickle as _pickle, types as _types
    cache_path = cached = None
    if problem_cache and not (layer_widths or depth or reask_queries or max_prob_tokens):
        cache_path = os.path.join(problem_cache, f"eval_op{max_op}_edge{max_edge}_exact{exact_op}"
                                                 f"_seed{seed}_n{num_problems}.pkl")
        if os.path.exists(cache_path):
            cached = _pickle.load(open(cache_path, "rb"))
            print(f"loaded {len(cached)} cached problems from {cache_path}", flush=True)
    to_cache = []
    for i in range(num_problems):
        hash_val = [TEST_HASH_BIN[i % len(TEST_HASH_BIN)]]
        while True:
            if cached is not None:
                pr, ptok, atok, stok = cached[i]
                id_gen = _types.SimpleNamespace(problem=pr, prob_token=ptok, ans_token=atok, sol_token=stok)
                reask_info = None
                break
            id_gen = GenClass(
                max_op=max_op,
                max_edge=max_edge,
                op=exact_op,
                perm_level=5,
                detail_level=0,
            )
            id_gen.gen_prob(hash_val, p_format="pq")
            # optional cap on problem length (tokens), so a geometry sweep
            # shifts only the distractor count, not the position range
            if max_prob_tokens and len(id_gen.prob_token) > max_prob_tokens:
                n_rejected_long += 1
                continue
            reask_info = _reask(id_gen, hash_val, reask_rng) if reask_queries else None
            if reask_queries and reask_info is None:
                continue
            break
        p = id_gen.problem
        if cache_path and cached is None:
            to_cache.append((p, list(id_gen.prob_token), list(id_gen.ans_token), list(id_gen.sol_token)))
            if (i + 1) % 64 == 0:
                print(f"  generating problems: {i + 1}/{num_problems}", flush=True)
        problems.append({
            "n_op": int(p.n_op),
            # the Problem object itself: parser verdicts are checked against this row's problem
            "problem_obj": p,
            "n_stated": len(p.problem) - 1,
            "n_needed": len(p.topological_order),
            "n_needed_stated": sum(1 for x in p.topological_order if x[0] == 0),
            "n_prob_tokens": len(id_gen.prob_token),
            "problem_struct": problem_structure(p, reask_info),
            "reask": (None if not reask_info else {
                "orig_op": reask_info["orig_op"], "new_op": reask_info["new_op"],
                "orig_query": p.get_ntn(param=reask_info["orig_query"]),
                "new_query": p.get_ntn(param=reask_info["new_query"])}),
            "problem_text": tokenizer.decode(id_gen.prob_token).strip(),
            "ground_truth": tokenizer.decode(id_gen.ans_token).strip(),
            "prompt": [TOKEN_EOS, TOKEN_START_PROBLEM]
                      + id_gen.prob_token + [TOKEN_START_SOLUTION],
        })

    if n_rejected_long:
        print(f"length cap {max_prob_tokens}: rejected {n_rejected_long} long problems")
    if cache_path and cached is None:
        os.makedirs(problem_cache, exist_ok=True)
        tmp = cache_path + f".tmp{os.getpid()}"
        _pickle.dump(to_cache, open(tmp, "wb")); os.replace(tmp, cache_path)
        print(f"cached {len(to_cache)} problems -> {cache_path}", flush=True)
    if build_only:
        return None

    def _generate(chunk):
        """Greedy generation for a list of prompts; left-padded when > 1."""
        if len(chunk) == 1:
            ids = torch.tensor([chunk[0]]).to(device)
            attn = torch.ones_like(ids)
            width = len(chunk[0])
        else:
            width = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), width), TOKEN_EOS, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            for r, ptoks in enumerate(chunk):
                ids[r, width - len(ptoks):] = torch.tensor(ptoks)
                attn[r, width - len(ptoks):] = 1
            ids, attn = ids.to(device), attn.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=TOKEN_EOS,
                pad_token_id=TOKEN_EOS,
                attention_mask=attn,
            )
        return [out[r][width:].tolist() for r in range(len(chunk))]

    generations = []
    for start in range(0, num_problems, gen_batch):
        chunk = [pb["prompt"] for pb in problems[start:start + gen_batch]]
        generations.extend(_generate(chunk))

    for i, (pb, gen_tokens) in enumerate(zip(problems, generations)):
        problem_text = pb["problem_text"]
        ground_truth = pb["ground_truth"]

        pred_sol = _extract_solution(gen_tokens)
        pred_ans = _extract_answer(gen_tokens)

        is_correct = _check_answer(pred_ans, ground_truth)
        stats = _parsing_stats(pred_sol, pb["problem_obj"])

        correct += int(is_correct)

        results.append(
            {
                "n_op": pb.get("n_op"),
                "n_stated": pb["n_stated"],
                "n_needed": pb["n_needed"],
                "n_needed_stated": pb["n_needed_stated"],
                "n_prob_tokens": pb["n_prob_tokens"],
                "problem_struct": pb["problem_struct"],
                "reask": pb["reask"],
                "problem": problem_text,
                "ground_truth": ground_truth,
                "predicted": pred_ans,
                "solution": pred_sol,
                "correct": is_correct,
                "op_setting": exact_op if exact_op else f"<= {max_op}",
                **stats,
            }
        )

        if (i + 1) % 10 == 0:
            acc = correct / (i + 1) * 100
            parse = sum(r["correct_parse"] for r in results) / (i + 1) * 100
            print(f"{i + 1}/{num_problems} | acc {acc:.1f}% | parse {parse:.1f}%")

    return {
        "accuracy": correct / num_problems * 100,
        "correct": correct,
        "total": num_problems,
        "results": results,
    }


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--difficulty", choices=["med"], default="med",
                        help="level preset; only 'med' (max_op 15, max_edge 20) is shipped")
    parser.add_argument("--num_problems", type=int, default=200)
    parser.add_argument("--igsm_root", default="./iGSM")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output_dir", default=None, help="Override output directory for results")
    parser.add_argument("--gen_batch", type=int, default=1,
                        help="problems per generate() call (left-padded batch)")
    parser.add_argument("--only_level", default=None,
                        help="run only levels whose name contains this string "
                             "(e.g. 'op<=15'); use for the bounded level, which "
                             "--only_op cannot select")
    parser.add_argument("--layer_widths", default=None,
                        help="W0,W1: override the generator's layer-width choice "
                             "(default 2-4 via softmax) to widen the problem world "
                             "(distractor sweep)")
    parser.add_argument("--max_edge", type=int, default=None,
                        help="override max_edge of every level (distractor sweep)")
    parser.add_argument("--problem_cache", default=os.environ.get("IGSM_EVAL_CACHE"),
                        help="directory of cached evaluation problems (generated once, shared by all models)")
    parser.add_argument("--build_cache_only", action="store_true",
                        help="generate and cache the problems of the selected levels, then exit (no model needed)")
    parser.add_argument("--reask", action="store_true",
                        help="re-point each problem's question at a random parameter "
                             "(the re-ask probe of Ye et al.); n_op becomes the new query's op")
    parser.add_argument("--depth", type=int, default=None,
                        help="force the number of category layers (default 2-4 via softmax)")
    parser.add_argument("--max_prob_tokens", type=int, default=0,
                        help="reject problems longer than this many tokens")
    parser.add_argument("--only_op", type=int, default=None,
                        help="run only the exact-op level with this max_op "
                             "(e.g. 20 for 'op=20 (OOD)'); the op<=N level is "
                             "never selected, so --only_op 15 gives op=15")

    args = parser.parse_args()

    EVAL_CONFIG["igsm_root"] = args.igsm_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.checkpoint or args.build_cache_only, "--checkpoint is required unless --build_cache_only"
    model = None if args.build_cache_only else _load_model(args.checkpoint, device)

    diff_level = EVAL_DIFFICULTIES_PRESETS[args.difficulty]

    if args.only_level is not None:
        levels = [d for d in diff_level["levels"] if args.only_level in d["name"]]
        if not levels:
            raise SystemExit(
                f"--only_level {args.only_level!r} matches nothing; available: "
                + ", ".join(d["name"] for d in diff_level["levels"]))
        diff_level = dict(diff_level, levels=levels)

    if args.only_op is not None:
        # "op<=15" and "op=15" share max_op, so require an exact-op level.
        levels = [d for d in diff_level["levels"]
                  if d["max_op"] == args.only_op and "<=" not in d["name"]]
        if not levels:
            raise SystemExit(
                f"--only_op {args.only_op} matches no exact-op level in "
                f"'{args.difficulty}'; available: "
                + ", ".join(d["name"] for d in diff_level["levels"]))
        diff_level = dict(diff_level, levels=levels)

    if args.max_edge is not None:
        diff_level = dict(diff_level, levels=[dict(d, max_edge=args.max_edge)
                                              for d in diff_level["levels"]])
    layer_widths = tuple(int(x) for x in args.layer_widths.split(",")) if args.layer_widths else None

    if args.output_dir:
        save_dir = args.output_dir
    elif args.checkpoint:
        ckpt_path = args.checkpoint.rstrip("/")
        ckpt_name = os.path.basename(ckpt_path)
        model_id = os.path.basename(os.path.dirname(ckpt_path)) if ckpt_name == "final_model" else ckpt_name
        save_dir = os.path.join("results", model_id, f"{args.difficulty}_n{args.num_problems}")
    else:
        save_dir = None   # --build_cache_only without a checkpoint writes no results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    for diff in diff_level["levels"]:
        print(f"\n{diff['name']}")

        exact_op = None if "<=" in diff["name"] else diff["max_op"]

        result = evaluate_difficulty(
            model,
            device,
            diff["max_op"],
            diff["max_edge"],
            args.num_problems,
            diff_level["max_new_tokens"],
            args.seed,
            exact_op,
            gen_batch=args.gen_batch,
            layer_widths=layer_widths,
            max_prob_tokens=args.max_prob_tokens,
            depth=args.depth,
            reask_queries=args.reask,
            problem_cache=args.problem_cache,
            build_only=args.build_cache_only,
        )

        if result is None:
            continue
        tag = f"op_{diff['max_op']}" if exact_op else f"op_le_{diff['max_op']}"
        path = os.path.join(save_dir, f"{tag}.json")

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"{diff['name']}: {result['accuracy']:.1f}%")

        all_results[diff["name"]] = result

    if args.build_cache_only:
        print("\nProblem cache built.")
        return

    # summary
    summary = {}

    for name, res in all_results.items():
        stats = res["results"]
        summary[name] = {
            "accuracy": res["accuracy"],
            "parse_rate": sum(r["correct_parse"] for r in stats) / len(stats) * 100,
            "incorrect_refer": sum(r["incorrect_refer"] for r in stats) / len(stats),
            "incorrect_calcs": sum(r["incorrect_calcs"] for r in stats) / len(stats),
            "incorrect_order": sum(r["incorrect_order"] for r in stats) / len(stats),
        }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary.")


if __name__ == "__main__":
    main()
