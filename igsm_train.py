import argparse
import glob
import random
import os
import shutil
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

CONFIG = {
    "seed": 42,
    "igsm_root" : "./iGSM",
    "max_op": 15,            # the med difficulty of iGSM
    "max_edge": 20,
    "train_steps": 100_000,
    "batch_size": 512,
    "grad_accum_steps":8,
    "learning_rate": 2e-3,
    "weight_decay": 0.05,
    "max_length": 768,
    "warmup_steps": 1_000,
    "save_steps": 2000,
    "logging_steps": 100,
    "output_dir": "./outputs/igsm_gpt2_output",
}


TOKEN_START_PROBLEM = 222
TOKEN_START_SOLUTION = 223
TOKEN_START_ANSWER = 224
TOKEN_EOS = 50256

# -- DDP helpers ----------

def setup_distributed():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def _add_igsm_to_path(cfg: dict):
    igsm_root = os.path.abspath(cfg["igsm_root"])
    if igsm_root not in sys.path:
        sys.path.insert(0, igsm_root)

def _apply_overrides(cfg: dict) -> dict:
    """Explicit CLI flags win over CONFIG."""
    merged = dict(cfg)
    merged.update(cfg.get("_overrides", {}))
    return merged

def _seed_everything(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    """Pad a batch of packed windows and build the labels: loss is taken on
    the [SOL] and [ANS] spans only (every token after a [SOL] or [ANS] marker
    up to and including the EOS); problem statements and padding are -100."""
    input_ids_list = [item["input_ids"] for item in batch]
    max_len = max(x.size(0) for x in input_ids_list)

    padded_input = torch.full((len(batch), max_len), TOKEN_EOS, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, inp in enumerate(input_ids_list):
        seq_len = inp.size(0)
        padded_input[i, :seq_len] = inp
        mask = torch.zeros(seq_len, dtype=torch.bool)
        supervising = False
        for j, tok in enumerate(inp.tolist()):
            if supervising:
                mask[j] = True
            if tok == TOKEN_START_SOLUTION or tok == TOKEN_START_ANSWER:
                supervising = True
            elif tok == TOKEN_START_PROBLEM or tok == TOKEN_EOS:
                supervising = False
        padded_labels[i, :seq_len][mask] = inp[mask]

    return {"input_ids": padded_input, "labels": padded_labels}

def _make_id_gen(cfg, redundant=None):
    """be_shortest=False emits a random topological order over the FULL
    template, truncated at the query: valid but non-minimal traces with a
    random amount of unnecessary computation. n_op still counts necessary
    ops only."""
    _add_igsm_to_path(cfg)
    sampling = cfg.get("sampling", "light")
    max_op, max_edge, op = cfg["max_op"], cfg["max_edge"], None
    if sampling == "light":
        from data_gen.pretrain.id_gen import IdGen
        return IdGen(max_op=max_op, max_edge=max_edge, op=op, perm_level=5, detail_level=0,
                     be_shortest=not (cfg.get("redundant_trace", False)
                                      if redundant is None else redundant))
    else:
        from data_gen.prototype.id_gen import IdGen_PT
        return IdGen_PT(sampling, sampling, max_op, max_edge, op=op, perm_level=5, detail_level=0,
                        be_shortest=not (cfg.get("redundant_trace", False)
                                         if redundant is None else redundant))


def _wandb_init(cfg, rank):
    """Optional wandb. Returns the module, or None when disabled/unavailable."""
    project = cfg.get("wandb_project")
    if not project or not is_main_process(rank):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb requested but not importable; continuing without it", flush=True)
        return None
    # wandb writes a ./wandb data directory, so when the package is absent that
    # directory becomes an empty namespace package and the import above quietly
    # succeeds. Catch it here rather than dying on wandb.init() mid-run.
    if not hasattr(wandb, "init"):
        print(f"wandb resolved to a stub ({wandb.__file__}); continuing without it",
              flush=True)
        return None
    wandb.init(project=project, name=cfg.get("wandb_run"),
               config={k: v for k, v in cfg.items() if not k.startswith("_")})
    return wandb


def _is_corrupted(cfg):
    """True when the training trace is altered (swap, no trace or prefix noise)."""
    return (cfg.get("swap_traces", False)
            or cfg.get("no_trace", False)
            or (cfg.get("prefix_noise_frac") or 0) > 0)


def _corruption_label(cfg):
    if cfg.get("swap_traces", False):
        return ("swapped traces (op-matched)" if cfg.get("swap_match_op")
                else "swapped traces")
    if cfg.get("no_trace", False):
        return "no trace (answer only)"
    if (cfg.get("prefix_noise_frac") or 0) > 0:
        if cfg["prefix_noise_frac"] >= 1.0:
            return "token-shuffled trace (entire trace, no valid remainder)"
        return f"token-shuffled trace prefix (first {cfg['prefix_noise_frac']:.0%} of sentences)"
    return "clean"


def drop_trace(id_gen):
    """Answer-only arm: remove the solution trace entirely, so the training
    sequence is [PROB] problem [SOL] [ANS] answer [EOS]. The problem and the
    answer are untouched. Returns 1 (one trace dropped)."""
    id_gen.sol_token = []
    id_gen.sol = ""
    id_gen.token_id = (
        [TOKEN_START_PROBLEM] + id_gen.prob_token +
        [TOKEN_START_SOLUTION] + id_gen.sol_token +
        [TOKEN_START_ANSWER] + id_gen.ans_token +
        [TOKEN_EOS]
    )
    return 1


def _maybe_corrupt(cfg, id_gen):
    """Apply the single-problem trace corruptions (the cross-problem swap is
    handled by the dataset, which needs a pair).

    Leaves the problem and the answer untouched, so the model is supervised on
    a correct answer reached through a missing or invalid derivation. Returns
    a count of what was altered (0 = clean).
    """
    if cfg.get("no_trace", False):
        return drop_trace(id_gen)

    pn = cfg.get("prefix_noise_frac") or 0
    if pn > 0:
        from traces.prefix_noise import noise_prefix
        return noise_prefix(id_gen, pn)

    return 0


# -- periodic validation ----------
# Held-out accuracy during training, scored on the final answer only: the
# altered-trace arms emit invalid or no traces by construction, so a parse-gated
# metric would read ~0 for them.



def _val_levels(cfg):
    """(name, max_op, max_edge, exact_op): the in-distribution range and op 20."""
    max_op, max_edge = cfg["max_op"], cfg["max_edge"]
    return [("in-dist", max_op, max_edge, None), ("ood_op20", 20, max_edge, 20)]


def _load_val_sets(cfg, n_per_level):
    """Validation sets from val_cache/: one JSON per level, shipped with the code.
    They were generated once with seed 1234 from the test hash bins (17-22), a
    fresh generator per problem, so every run and every arm sees the same problems."""
    import json as _json
    cache_dir = os.environ.get("IGSM_VAL_CACHE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "val_cache")

    def _prompt(prob_token):
        return [TOKEN_EOS, TOKEN_START_PROBLEM] + list(prob_token) + [TOKEN_START_SOLUTION]

    sets = []
    for name, max_op, max_edge, exact_op in _val_levels(cfg):
        key = f"val_v2_op{max_op}_edge{max_edge}_exact{exact_op}_seed1234.json"
        path = os.path.join(cache_dir, key)
        if not os.path.exists(path):
            raise SystemExit(f"validation set {name}: {path} not found")
        cached = _json.load(open(path))["items"]
        if len(cached) < n_per_level:
            raise SystemExit(f"validation set {name}: {key} holds {len(cached)} problems, "
                             f"--eval_problems asks for {n_per_level}")
        raw = cached[:n_per_level]
        print(f"validation set {name}: {n_per_level} problems loaded from cache "
              f"({key}, {len(cached)} stored)", flush=True)
        sets.append((name, [(_prompt(r["prob_token"]), r["answer"]) for r in raw]))
    return sets


def _answer_of(generated, tokenizer):
    """Decode the answer segment of a generation, mirroring evaluate.py."""
    if TOKEN_START_ANSWER not in generated:
        return ""
    tail = generated[generated.index(TOKEN_START_ANSWER) + 1:]
    if TOKEN_EOS in tail:
        tail = tail[: tail.index(TOKEN_EOS)]
    return tokenizer.decode(tail).strip()


# Matches evaluate.py: the swapped arm emits long outputs and would be cut off
# before [ANS] at a smaller limit.
VAL_MAX_NEW_TOKENS = 2048


def _run_validation(model, val_sets, device, tokenizer, max_new_tokens=VAL_MAX_NEW_TOKENS,
                    print_n=0, step=None, batch_size=32):
    """Greedy generation over the validation sets, batched.

    Prompts are left-padded so every row ends at the same index; the attention
    mask zeroes the padding and generate() derives position ids from it, so a
    row's output is independent of what it is batched with.
    """
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for name, items in val_sets:
            correct = 0
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                width = max(len(p) for p, _ in chunk)
                ids = torch.full((len(chunk), width), TOKEN_EOS, dtype=torch.long)
                attn = torch.zeros((len(chunk), width), dtype=torch.long)
                for i, (prompt, _) in enumerate(chunk):
                    ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
                    attn[i, width - len(prompt):] = 1
                gen = model.generate(
                    ids.to(device),
                    attention_mask=attn.to(device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=TOKEN_EOS,
                    pad_token_id=TOKEN_EOS,
                )
                for i, (prompt, gt) in enumerate(chunk):
                    produced = gen[i][width:].tolist()
                    pred = _answer_of(produced, tokenizer)
                    correct += int(pred == gt)
                    idx = start + i
                    if idx < print_n:
                        tag = "OK " if pred == gt else "BAD"
                        print(f"  [VALGEN step {step} {name} #{idx+1} {tag}] "
                              f"pred={pred!r} gt={gt!r}", flush=True)
                        print("    " + tokenizer.decode(produced).strip()[:600], flush=True)
            out.append((name, 100.0 * correct / max(1, len(items))))
    if was_training:
        model.train()
    return out


def test(cfg:dict):
    cfg = _apply_overrides(cfg)
    _add_igsm_to_path(cfg)
    from tools.tools import tokenizer, fix_seed

    max_op = cfg["max_op"]
    max_edge = cfg["max_edge"]
    fix_seed(cfg["seed"])

    print(f"\n{'='*60}")
    print(f"TEST MODE - 5 problems (max_op={cfg['max_op']}, sampling={cfg.get('sampling', 'light')})")
    print(f"\n{'='*60}")

    all_bins = list(range(23))
    for i in range(5):
        id_gen = _make_id_gen(cfg)
        id_gen.gen_prob(all_bins, p_format="pq")
        n_altered = _maybe_corrupt(cfg, id_gen)

        full_tokens = id_gen.token_id
        assert full_tokens == (
            [TOKEN_START_PROBLEM] + 
            id_gen.prob_token +
            [TOKEN_START_SOLUTION] +
            id_gen.sol_token +
            [TOKEN_START_ANSWER] +
            id_gen.ans_token + 
            [TOKEN_EOS]
        )

        print(f"Example {i+1}" + (f"  [{_corruption_label(cfg)} {n_altered}]" if n_altered else ""))
        print(f"Problem: {tokenizer.decode(id_gen.prob_token).strip()}")
        print(f"Solution: {tokenizer.decode(id_gen.sol_token).strip()}")
        print(f"Answer: {tokenizer.decode(id_gen.ans_token).strip()}")
        print(f"Tokens (len): {len(full_tokens)}")
        print()
    print("iGSM is working. Run --mode train to start training.\n")


class IGSMIterableDataset(IterableDataset):
    TRAIN_HASH_BIN = list(range(17))

    def __init__(self, cfg: dict, rank: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.cfg["seed"] + self.rank * 1000 + worker_info.id
        else:
            seed = self.cfg["seed"] + self.rank * 1000

        _add_igsm_to_path(self.cfg)
        from tools.tools import fix_seed

        fix_seed(seed)

        max_length = self.cfg["max_length"]

        buffer = []

        swap = self.cfg.get("swap_traces", False)
        if swap:
            from traces.swap_trace import swap_solutions

        match_op = self.cfg.get("swap_match_op", False)
        pool = {}   # n_op -> waiting problem (op-matched swap only)

        while True:
            if swap:
                # Cross-problem swap needs a pair. If either swapped sequence
                # overflows the window, discard the pair -- same skip-the-long
                # rule as the single-problem path, applied jointly so the
                # pairing never mixes with an unswapped leftover.
                if match_op:
                    # Pair each problem with one of the SAME op count, so the
                    # supervised trace length scales with problem depth and
                    # length is deconfounded from content (the trace is still
                    # another problem's). Problems wait in a per-op pool until
                    # a partner arrives; each is used exactly once.
                    a = _make_id_gen(self.cfg)
                    a.gen_prob(self.TRAIN_HASH_BIN, p_format="pq")
                    op = a.problem.n_op
                    if op not in pool:
                        pool[op] = a
                        continue
                    b = pool.pop(op)
                else:
                    a = _make_id_gen(self.cfg)
                    a.gen_prob(self.TRAIN_HASH_BIN, p_format="pq")
                    b = _make_id_gen(self.cfg)
                    b.gen_prob(self.TRAIN_HASH_BIN, p_format="pq")
                swap_solutions(a, b)
                if len(a.token_id) > max_length or len(b.token_id) > max_length:
                    continue
                emitted = [a.token_id, b.token_id]
            else:
                frac = self.cfg.get("redundant_frac")
                if frac is not None:
                    # per-problem mix: redundant with probability frac, minimal otherwise
                    id_gen = _make_id_gen(self.cfg,
                                          redundant=random.random() < frac)
                else:
                    id_gen = _make_id_gen(self.cfg)
                id_gen.gen_prob(self.TRAIN_HASH_BIN, p_format="pq")
                _maybe_corrupt(self.cfg, id_gen)
                if len(id_gen.token_id) > max_length:
                    continue
                emitted = [id_gen.token_id]

            for tokens in emitted:
                buffer.extend(tokens)
                if len(buffer) >= max_length:
                    # tokens past the window are dropped: the last problem of a window
                    # is truncated and the next window starts at a fresh [PROB]
                    yield {"input_ids": torch.tensor(buffer[:max_length], dtype=torch.long)}
                    buffer = []

from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup_and_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

def _find_resume_checkpoint(output_dir):
    """Latest checkpoint-N holding a training_state.pt, or None."""
    cands = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if os.path.exists(os.path.join(d, "training_state.pt")):
            cands.append((int(d.split("-")[-1]), d))
    return max(cands)[1] if cands else None


def train(cfg: dict):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    cfg = _apply_overrides(cfg)

    _add_igsm_to_path(cfg)
    from tools.tools import tokenizer, fix_seed
    fix_seed(999)
    sample_id_gen = _make_id_gen(cfg)
    sample_id_gen.gen_prob(list(range(17, 23)), p_format="pq")
    sample_gt = tokenizer.decode(sample_id_gen.ans_token).strip()
    sample_prompt = [TOKEN_EOS, TOKEN_START_PROBLEM] + sample_id_gen.prob_token + [TOKEN_START_SOLUTION]
    sample_problem_text = tokenizer.decode(sample_id_gen.prob_token).strip()

    wb = _wandb_init(cfg, rank)

    val_sets = None
    if is_main_process(rank) and cfg.get("eval_steps", 0) > 0:
        t_val = time.time()
        val_sets = _load_val_sets(cfg, cfg.get("eval_problems", 32))
        print("validation sets: " + ", ".join(
            f"{n}={len(items)}" for n, items in val_sets)
            + f" (loaded in {time.time() - t_val:.0f}s)", flush=True)

    _seed_everything(cfg["seed"])

    # each GPU processes batch_size / (world_size * grad_accum_steps) samples per forward pass
    assert cfg["batch_size"] % (cfg["grad_accum_steps"] * world_size) == 0, (
        f"batch_size ({cfg['batch_size']}) must be divisible by "
        f"grad_accum_steps * world_size ({cfg['grad_accum_steps']} * {world_size})"
    )
    micro_batch_size = cfg["batch_size"] // (cfg["grad_accum_steps"] * world_size)
    dataset = IGSMIterableDataset(cfg, rank=rank)

    num_workers = 6
    train_loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=4,
    )

    if is_main_process(rank):
        print(f"\nBuilding GPT-2 sized GPTNeoX with RoPE from scratch (random_weights)...")
    config = GPTNeoXConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        rotary_pct=1.0,
        vocab_size=50257,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        use_parallel_residual=False,
    )
    model = GPTNeoXForCausalLM(config)

    if is_main_process(rank):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Micro batch size per GPU: {micro_batch_size}")
        # State the trace treatment explicitly. A corruption flag that fails to
        # reach the job trains a silently clean arm that looks identical in the
        # log, and the mistake only surfaces at eval time.
        if _is_corrupted(cfg):
            print(f"Trace corruption: {_corruption_label(cfg)}")
        else:
            print("Trace corruption: none (clean)")
        if cfg.get("redundant_frac") is not None:
            print(f"Trace style: redundant with p={cfg['redundant_frac']}, "
                  "minimal otherwise (per-problem mix)")
        elif cfg.get("redundant_trace"):
            print("Trace style: redundant (be_shortest=False, random "
                  "non-minimal topological order)")

    resume_ckpt = _find_resume_checkpoint(cfg["output_dir"]) if cfg.get("resume") else None
    resume_step = 0
    if resume_ckpt:
        from safetensors.torch import load_file
        state = load_file(os.path.join(resume_ckpt, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        # the tied output head (embed_out, tied to embed_in) is not stored in
        # the file; anything else missing is a bug
        assert not unexpected and all(k == "embed_out.weight" for k in missing), \
            (missing, unexpected)
        resume_step = int(resume_ckpt.split("-")[-1])
        if is_main_process(rank):
            print(f"Resuming from {resume_ckpt} (step {resume_step})", flush=True)

    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.98)
    )
    scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer, 
        num_warmup_steps=cfg["warmup_steps"], 
        num_training_steps=cfg["train_steps"]
    )

    if is_main_process(rank):
        print(f"\nTraining for {cfg['train_steps']:,} steps "
             f"(effective batch size {cfg['batch_size']}, "
             f"grad accum {cfg['grad_accum_steps']}, "
             f"sampling {cfg.get('sampling', 'light')}), "
             f"GPUs {world_size}...\n")
    if resume_ckpt:
        ts = torch.load(os.path.join(resume_ckpt, "training_state.pt"),
                        map_location=device, weights_only=True)
        assert ts["global_step"] == resume_step
        optimizer.load_state_dict(ts["optimizer"])
        scheduler.load_state_dict(ts["scheduler"])
        # The data stream is procedural and infinite; replaying it to the exact
        # position is not supported. Move the seed (the loader workers read it
        # from cfg when they start) so the continued run draws fresh samples
        # instead of repeating the beginning.
        cfg["seed"] += 100_000 + resume_step
        _seed_everything(cfg["seed"])

    model.train()
    global_step = resume_step
    accum_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for batch in train_loader:
        if global_step >= cfg["train_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, labels = labels)
            loss = outputs.loss / cfg["grad_accum_steps"]

        loss.backward()
        running_loss += loss.item()
        accum_step += 1
        if accum_step %cfg["grad_accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main_process(rank) and global_step % cfg["logging_steps"] == 0:
                avg_loss = running_loss / cfg["logging_steps"]
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                done = global_step - resume_step
                eta_hours = (cfg["train_steps"] - global_step) / (done / elapsed) / 3600 if done > 0 else 0
                print(f"step {global_step:6d}/{cfg['train_steps']} | "
                      f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"elapsed {elapsed:.0f}s | ETA {eta_hours:.1f}h")
                if wb:
                    wb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

                unwrapped_model = model.module if world_size > 1 else model
                unwrapped_model.eval()

                with torch.no_grad():
                    prompt_ids = torch.tensor([sample_prompt], dtype=torch.long).to(device)
                    attention_mask = torch.ones_like(prompt_ids)
                    output_ids = unwrapped_model.generate(
                        prompt_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        do_sample=False,
                        eos_token_id=TOKEN_EOS,
                        pad_token_id=TOKEN_EOS,
                    )
                    generated = output_ids[0][len(sample_prompt):].tolist()
                    generated_text = tokenizer.decode(generated).strip()
                    print(f"problem: {sample_problem_text}")
                    print(f"generated: {generated_text}")
                    print(f"gt_answer: {sample_gt}")
                model.train()

            if global_step % cfg["logging_steps"] == 0:
                running_loss = 0.0

            if (val_sets is not None
                    and global_step > 0
                    and global_step % cfg["eval_steps"] == 0):
                t_val = time.time()
                unwrapped_model = model.module if world_size > 1 else model
                scores = _run_validation(unwrapped_model, val_sets, device, tokenizer,
                                         print_n=cfg.get('val_print', 0),
                                         step=global_step)
                print("VAL step {:6d} | ".format(global_step)
                      + " | ".join(f"{n} {a:.1f}%" for n, a in scores)
                      + f" | {time.time() - t_val:.0f}s", flush=True)
                if wb:
                    wb.log({f"val/{n}": a for n, a in scores}, step=global_step)

            if global_step % cfg["save_steps"] == 0:
                if is_main_process(rank):
                    unwrapped_model = model.module if world_size > 1 else model
                    ckpt_path = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                    unwrapped_model.save_pretrained(ckpt_path)
                    torch.save({"global_step": global_step,
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict()},
                               os.path.join(ckpt_path, "training_state.pt"))
                    print(f"Saved checkpoint {ckpt_path}")

                    test_model = GPTNeoXForCausalLM.from_pretrained(ckpt_path).to(device).eval()
                    with torch.no_grad():
                        prompt_ids = torch.tensor([sample_prompt], dtype=torch.long).to(device)
                        output_ids = test_model.generate(
                            prompt_ids,
                            attention_mask=torch.ones_like(prompt_ids),
                            max_new_tokens=256,
                            do_sample=False,
                            eos_token_id=TOKEN_EOS,
                            pad_token_id=TOKEN_EOS,
                        )
                        generated = output_ids[0][len(sample_prompt):].tolist()
                        print(f"Reloaded generated:  {tokenizer.decode(generated).strip()}")
                        print(f"gt_answer: {sample_gt}")
                    del test_model

                    # Remove old checkpoints, keep last N
                    keep_n = 3
                    ckpts = sorted(
                        glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")),
                        key=lambda x: int(x.split("-")[-1])
                    )
                    for old in ckpts[:-keep_n]:
                        print(f"Removing old checkpoint {old}")
                        shutil.rmtree(old)

                if world_size > 1:
                    dist.barrier()

    if is_main_process(rank):
        unwrapped_model = model.module if world_size > 1 else model
        final_path = os.path.join(cfg["output_dir"], "final_model")
        unwrapped_model.save_pretrained(final_path)
        print(f"\nTraining complete. Final model saved to {final_path}")

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 (GPTNeoX+RoPE) from scratch on iGSM math problems")
    parser.add_argument("--mode", choices=["test", "train"], default="train",
                        help="test=print five generated training examples and exit | train=train")
    parser.add_argument("--sampling", choices=["light", "uniform", "middle", "heavy"], default="light",
                        help="Problem sampling distribution: light (biased easier), uniform (flat), middle, heavy (biased harder)")
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help="Run held-out validation every N steps (0 disables)")
    parser.add_argument("--eval_problems", type=int, default=32,
                        help="Problems per validation level")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Log to this wandb project (omit to disable wandb entirely)")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="wandb run name")
    parser.add_argument("--max_length", type=int, default=None,
                        help="context window (default 768)")
    parser.add_argument("--swap_traces", action="store_true",
                        help="supervise each problem with another problem's "
                             "solution trace (answers stay correct) -- the "
                             "Beyond Semantics swapped-trace protocol")
    parser.add_argument("--no_trace", action="store_true",
                        help="answer-only arm: drop the solution trace entirely, "
                             "so the training sequence is [PROB] problem [SOL] "
                             "[ANS] answer [EOS]")
    parser.add_argument("--redundant_frac", type=float, default=None,
                        help="per-problem probability of a redundant "
                             "(be_shortest=False) trace; the rest stay "
                             "minimal (0.9 = the 90/10 mix)")
    parser.add_argument("--redundant_trace", action="store_true",
                        help="train on valid but non-minimal traces "
                             "(be_shortest=False): a random topological order "
                             "over the full template, truncated at the query")
    parser.add_argument("--swap_match_op", action="store_true",
                        help="with --swap_traces: partner problems by equal op "
                             "count, so trace length is deconfounded from "
                             "trace content")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in output_dir "
                             "that has a training_state.pt (optimizer+scheduler "
                             "+step); the data stream is re-seeded, not replayed")
    parser.add_argument("--prefix_noise_frac", type=float, default=None,
                        help="permute all tokens of the first FRAC of the solution sentences "
                             "(at least 1, at most n-1; 1.0 = the whole trace); the rest of the "
                             "trace, the problem and the answer are untouched")
    parser.add_argument("--val_print", type=int, default=0,
                        help="Print the first N validation generations at each eval (0 = off)")
    parser.add_argument("--igsm_root", type=str, default=None,
                        help="path to the iGSM generator checkout (default ./iGSM)")
    parser.add_argument("--train_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed: model init and the problem stream. The "
                             "validation sets are fixed separately, so runs at different "
                             "seeds stay comparable on the same held-out problems.")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="gradient accumulation steps; per-GPU micro-batch = batch_size // (grad_accum_steps * world_size)")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    args = parser.parse_args()

    cfg= dict(CONFIG)
    cfg["sampling"] = args.sampling
    cfg["eval_steps"] = args.eval_steps
    cfg["eval_problems"] = args.eval_problems
    cfg["val_print"] = args.val_print
    cfg["swap_traces"] = args.swap_traces
    cfg["no_trace"] = args.no_trace
    n_corrupt = sum([bool(args.swap_traces), bool(args.no_trace),
                     (args.prefix_noise_frac or 0) > 0])
    assert n_corrupt <= 1, "trace corruptions are mutually exclusive"
    if args.prefix_noise_frac is not None:
        assert 0.0 < args.prefix_noise_frac <= 1.0, "--prefix_noise_frac must be in (0, 1]"
        assert not (args.redundant_trace or args.redundant_frac is not None), \
            "--prefix_noise_frac applies to the minimal trace only"
    cfg["prefix_noise_frac"] = args.prefix_noise_frac
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.swap_match_op:
        assert args.swap_traces, "--swap_match_op requires --swap_traces"
    cfg["swap_match_op"] = args.swap_match_op
    if args.redundant_trace:
        assert not (args.swap_traces or args.no_trace), \
            "--redundant_trace is a trace-style variant, exclusive of corruptions"
    cfg["redundant_trace"] = args.redundant_trace
    if args.redundant_frac is not None:
        assert 0.0 <= args.redundant_frac <= 1.0
        assert not args.redundant_trace, \
            "--redundant_frac subsumes --redundant_trace (use one)"
        assert not (args.swap_traces or args.no_trace), \
            "--redundant_frac is a trace-style variant, exclusive of corruptions"
    cfg["redundant_frac"] = args.redundant_frac
    cfg["resume"] = args.resume
    cfg["wandb_project"] = args.wandb_project
    cfg["wandb_run"] = args.wandb_run
    if args.igsm_root is not None: cfg["igsm_root"] = args.igsm_root
    if args.train_steps is not None: cfg["train_steps"] = args.train_steps
    if args.grad_accum_steps is not None: cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.output_dir is not None: cfg["output_dir"] = args.output_dir

    # flags that must win over CONFIG are re-applied last by _apply_overrides()
    cfg["_overrides"] = {
        k: v for k, v in (
            ("train_steps", args.train_steps),
            ("grad_accum_steps", args.grad_accum_steps),
            ("max_length", args.max_length),
        ) if v is not None
    }

    assert cfg["batch_size"] % cfg["grad_accum_steps"] == 0, (
        f"batch_size ({cfg['batch_size']}) must be divisible by "
        f"grad_accum_steps ({cfg['grad_accum_steps']})"
    )
    if args.mode == "test":
        test(cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()
