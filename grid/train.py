import argparse
import os
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainingArguments

from grid.alignment import AlignmentHead, VGGTTeacher, alignment_loss, visual_hidden_states
from grid.data import Collator, NavigationDataset
from grid.model import LORA_TARGETS, OPENVLA, OPENVLA_REVISION, VISUAL_TOKENS, build_policy

HEAD_FILE = "alignment_head.safetensors"
ALIGNMENT_WEIGHT = 0.5


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


class GRIDTrainer(Trainer):
    def __init__(self, *args, teacher, alignment_weight=ALIGNMENT_WEIGHT, student_layer=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.alignment_weight = alignment_weight
        self.student_layer = student_layer

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        views = inputs.pop("teacher_views")
        with torch.no_grad():
            targets = self.teacher(views)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        features = visual_hidden_states(outputs, self.student_layer, VISUAL_TOKENS)
        aligned = unwrap(model).alignment_head(features)
        loss = outputs.loss.float() + self.alignment_weight * alignment_loss(aligned, targets)
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.args.should_save:
            folder = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            state = {k: v.detach().cpu().contiguous() for k, v in unwrap(self.model).alignment_head.state_dict().items()}
            save_file(state, str(folder / HEAD_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        path = Path(resume_from_checkpoint) / HEAD_FILE
        if path.exists():
            unwrap(model if model is not None else self.model).alignment_head.load_state_dict(load_file(str(path)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=OPENVLA)
    parser.add_argument("--revision", default=OPENVLA_REVISION)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--depth-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-layer", type=int, default=24)
    parser.add_argument("--vggt-checkpoint", default=None)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    accumulation, remainder = divmod(args.global_batch, world_size * args.micro_batch)
    if remainder or accumulation < 1:
        raise SystemExit("--global-batch must be a multiple of GPUs x --micro-batch")

    transformers.set_seed(42)
    model, tokenizer, processor = build_policy(args.model_path, device_map={"": local_rank}, revision=args.revision)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGETS), modules_to_save=["projector"],
    ))
    device = torch.device("cuda", local_rank)
    model.add_module("alignment_head", AlignmentHead().to(device))
    with torch.random.fork_rng(devices=[local_rank]):
        teacher = VGGTTeacher(args.vggt_checkpoint, device=device)
    model.print_trainable_parameters()

    dataset = NavigationDataset(args.manifest, args.data_root, tokenizer, processor, args.depth_cache)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.03,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_torch",
        dataloader_num_workers=args.workers,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=args.report_to,
        seed=42,
    )
    trainer = GRIDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(tokenizer),
        teacher=teacher,
        alignment_weight=ALIGNMENT_WEIGHT,
        student_layer=args.student_layer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    final = Path(args.output_dir) / "final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final))


if __name__ == "__main__":
    main()
