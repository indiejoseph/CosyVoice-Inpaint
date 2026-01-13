import os
import sys
import torch  # PyTorch library for deep learning
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_utils import EvalLoopOutput
from transformers import (
    Trainer,
    TrainingArguments,  # Training arguments for model training
    AutoTokenizer,
    Qwen2ForCausalLM,
    AutoConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from torch.nn.utils.rnn import pad_sequence, unpad_sequence

# from peft import LoraConfig, get_peft_model

sys.path.append("/home/pj24001684/ku40000295/jc/projects/CosyVoice")

from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.utils.common import ras_sampling, th_accuracy
from cosyvoice.utils.mask import make_pad_mask

# The model that you want to train from the Hugging Face hub
qwen_pretrain_model = "qwen_pretrained/blank"

# Fine-tuned model name
new_model = "cosyvoice2-2025090507"
num_speech_tokens = 6561

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = f"results/{new_model}"
os.makedirs("results", exist_ok=True)

# Number of training epochs
num_train_epochs = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 32

# Batch size per GPU for evaluation
per_device_eval_batch_size = 16

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 1e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01

# Optimizer to use
optim = "paged_adamw_8bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "cosine_with_min_lr"
lr_scheduler_kwargs = {"min_lr": 1.0e-7}
# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.1
# warmup_steps = 2000

# Log every X updates steps
logging_steps = 100

# Load the entire model on the GPU 0
device = "cuda:0"

# Step 1 : Load dataset (you can process it here)
df = pd.read_csv("dataset20250907.csv")
ds = Dataset.from_pandas(df, split="train")


def convert_to_list(x, offset=0):
    return [int(i) + offset for i in x.split(" ")]


tokenizer = AutoTokenizer.from_pretrained(qwen_pretrain_model)


def tokenize_add_label(sample):
    text = tokenizer.encode(sample["text"], add_special_tokens=True)
    speech_token = convert_to_list(sample["codes"])

    sample = {
        "text": text,
        "speech_token": speech_token,
    }

    return sample


dataset = ds.map(tokenize_add_label, remove_columns=list(ds.features))
dataset = dataset.shuffle(seed=42)

# split the dataset into training and validation
ds = dataset.train_test_split(test_size=1000 / len(df))
train_ds = ds["train"]
eval_ds = ds["test"]

# print first row of the dataset
print("text", train_ds[0]["text"])
print("speech_token", train_ds[0]["speech_token"])


def cosyvoice2_forward(
    self,
    text_token: torch.Tensor,
    text_token_len: torch.Tensor,
    speech_token: torch.Tensor,
    speech_token_len: torch.Tensor,
) -> Dict[str, Optional[torch.Tensor]]:

    # 1. encode text_token
    text_token_emb = self.llm.model.model.embed_tokens(text_token)

    # 2. encode speech_token
    speech_token_emb = self.speech_embedding(speech_token)

    # 3. prepare llm_input/target
    lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
        text_token,
        text_token_emb,
        text_token_len,
        speech_token,
        speech_token_emb,
        speech_token_len,
    )
    lm_target = lm_target.to(device)

    # 4. run lm forward
    lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
    logits = self.llm_decoder(lm_output)
    loss = self.criterion_ce(logits, lm_target.to(device))
    acc = th_accuracy(
        logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID
    )
    return {"loss": loss, "acc": acc}


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, config_file: str):
        super().__init__()
        config = AutoConfig.from_pretrained(config_file)
        self.model = Qwen2ForCausalLM(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


# Step 2 :Load base model
# peft_config = LoraConfig(
#     r=32,
#     lora_alpha=64,
#     lora_dropout=0.05,
#     target_modules=[
#         "up_proj",
#         "gate_proj",
#         "o_proj",
#         "down_proj",
#         "v_proj",
#         "q_proj",
#         "k_proj",
#     ],
#     modules_to_save=["lm_head", "embed_token"],
#     task_type="CAUSAL_LM",
# )

llm = Qwen2Encoder(qwen_pretrain_model)
model = Qwen2LM(
    llm_input_size=896,
    llm_output_size=896,
    speech_token_size=num_speech_tokens,
    llm=llm,
    length_normalized_loss=True,
    lsm_weight=0.0,
    sampling=ras_sampling,
).to(device)

# Load the model weights
model.load_state_dict(
    torch.load(
        "qwen_pretrained/llm.pt",
        weights_only=True,
    )
)

# Apply the PEFT model
# llm.model = get_peft_model(llm.model, peft_config)

# Print trainable parameters
# llm.model.print_trainable_parameters()

# Override model.forward to cosyvoice2_forward
model.forward = cosyvoice2_forward.__get__(model)

# if torch.__version__ >= "2" and sys.platform != "win32":
#     llm.model = torch.compile(llm.model)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    auto_find_batch_size=False,
    eval_strategy="epoch",
    optim=optim,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    save_strategy="epoch",
    save_safetensors=False,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=False,
    lr_scheduler_type=lr_scheduler_type,
    lr_scheduler_kwargs=lr_scheduler_kwargs,
    remove_unused_columns=False,
    label_names=["speech_token"],
    report_to="wandb",
)


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_acc = 0.0  # Initialize best accuracy to 0
        self.best_models = []  # List to store (acc, path) tuples, highest acc first
        # Create the output directory if it doesn't exist
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only save models on the main process
        if not args.local_rank in [-1, 0]:
            return

        metrics = kwargs.get("metrics", {})
        eval_acc = metrics.get("eval_accuracy")  # Get accuracy from metrics

        if eval_acc is not None:
            best_model_dir = os.path.join(self.output_dir, "best_model")

            # Use args.save_total_limit as the number of models to keep
            top_k = (
                args.save_total_limit
                if hasattr(args, "save_total_limit")
                and args.save_total_limit is not None
                else 1
            )

            # Save model with accuracy value in filename
            model_path = os.path.join(best_model_dir, f"model_acc_{eval_acc:.4f}.pt")

            # Check if this model should be in top-k
            should_save = False

            if len(self.best_models) < top_k:
                # Less than k models saved, save this one
                should_save = True
                improvement = (
                    eval_acc - self.best_acc if eval_acc > self.best_acc else 0
                )
            elif eval_acc > self.best_models[-1][0]:  # Better than worst model in top-k
                # Remove the worst model file if it exists
                worst_model_path = self.best_models[-1][1]
                if os.path.exists(worst_model_path):
                    try:
                        os.remove(worst_model_path)
                    except OSError as e:
                        print(f"Error removing old model file: {e}")

                # Remove worst model from list
                self.best_models.pop()
                should_save = True
                improvement = (
                    eval_acc - self.best_models[-1][0] if self.best_models else 0
                )

            if should_save:
                # Save the model
                torch.save(kwargs["model"].state_dict(), model_path)

                # Add to our sorted list of best models
                self.best_models.append((eval_acc, model_path))
                # Sort by accuracy (descending)
                self.best_models.sort(key=lambda x: x[0], reverse=True)

                # If this is the best model so far, update best_acc and save as best_model.pt
                if eval_acc > self.best_acc:
                    self.best_acc = eval_acc
                    best_model_path = os.path.join(best_model_dir, "best_model.pt")
                    torch.save(kwargs["model"].state_dict(), best_model_path)
                    print(
                        f"\n✅ New best model saved! Accuracy improved by {improvement:.6f} to {eval_acc:.6f}"
                    )
                else:
                    print(
                        f"\n✅ Model added to top-{top_k}! Current accuracy: {eval_acc:.6f}, Best accuracy: {self.best_acc:.6f}"
                    )

                # Print out all saved models
                print(f"Current top-{len(self.best_models)} models:")
                for i, (acc, path) in enumerate(self.best_models):
                    print(f"  {i+1}. Accuracy: {acc:.6f} - {os.path.basename(path)}")
            else:
                print(
                    f"\n⏺️ No improvement for top-{top_k}. Current accuracy: {eval_acc:.6f}, Worst top-k: {self.best_models[-1][0]:.6f}"
                )
        else:
            print("\n⚠️ No evaluation accuracy available.")


class CustomDataCollatorWithPadding:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "text_token": pad_sequence(
                [torch.LongTensor(f["text"]) for f in features],
                batch_first=True,
                padding_value=151643,
            ),
            "text_token_len": torch.LongTensor([len(f["text"]) for f in features]),
            "speech_token": pad_sequence(
                [torch.LongTensor(f["speech_token"]) for f in features],
                batch_first=True,
                padding_value=num_speech_tokens,
            ),
            "speech_token_len": torch.LongTensor(
                [len(f["speech_token"]) for f in features]
            ),
        }
        return batch


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)

        # Assuming outputs is a dictionary with 'loss' and 'acc' keys
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            # Log the accuracy during training
            if "acc" in outputs and self.args.logging_steps > 0:
                # Only log if current step is a logging step
                if self.state.global_step % self.args.logging_steps == 0:
                    self.log(
                        {"training_accuracy": outputs["acc"].detach().float().item()}
                    )
        else:
            loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Call the parent class's evaluation loop
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Get the metrics from the parent class
        metrics = (
            eval_output.metrics.copy()
        )  # Create a copy to avoid modifying the original

        # Calculate accuracy across the entire evaluation dataset
        total_acc = 0.0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = self._prepare_inputs(batch)
                outputs = self.model(**batch)
                if isinstance(outputs, dict) and "acc" in outputs:
                    # Add batch accuracy to total, weighted by batch size
                    total_acc += outputs["acc"].item() * len(batch["text_token"])
                    total_samples += len(batch["text_token"])

        # Add the accuracy to metrics
        if total_samples > 0:
            metrics[f"{metric_key_prefix}_accuracy"] = total_acc / total_samples

        # Create a new EvalLoopOutput with updated metrics
        new_eval_output = EvalLoopOutput(
            predictions=eval_output.predictions,
            label_ids=eval_output.label_ids,
            metrics=metrics,
            num_samples=eval_output.num_samples,
        )

        return new_eval_output


data_collator = CustomDataCollatorWithPadding()

# test data_collator
print(data_collator([train_ds[0], train_ds[1]]))

trainer = CustomTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[SaveBestModelCallback(output_dir)],
)

# Train model
trainer.train()

# save model
trainer.save_model()
