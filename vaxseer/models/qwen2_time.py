import torch
import pytorch_lightning as pl
import transformers
import torch.nn as nn
from data.utils import discretize_time
from torch.nn import CrossEntropyLoss
from models import register_model
import math, logging
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union
from utils.args import str2bool
from transformers import AutoConfig


# --- qwen2 building blocks (pure pytorch) ---

class RMSNorm(nn.Module):
    """root mean square layer normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """rotary position embedding"""
    def __init__(self, dim, max_position_embeddings=1024, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


class SwiGLU(nn.Module):
    """swiglu feedforward"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """grouped query attention"""
    def __init__(self, hidden_size, num_attention_heads, num_kv_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x, cos, sin, attention_mask=None):
        B, L, _ = x.size()

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # expand kv for grouped query attention
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L, self.head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # use a large finite negative value, not -inf, to stay fp16-safe
        neg_fill = torch.finfo(attn_weights.dtype).min

        # causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg_fill)

        if attention_mask is not None:
            # attention_mask: [B, L], 1 = attend, 0 = mask
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B,1,1,L]
            attn_weights = attn_weights.masked_fill(~pad_mask, neg_fill)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # guard against any all-masked row producing nan
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2Block(nn.Module):
    """single transformer block"""
    def __init__(self, hidden_size, num_attention_heads, num_kv_heads, intermediate_size):
        super().__init__()
        self.self_attn = Qwen2Attention(hidden_size, num_attention_heads, num_kv_heads)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x, cos, sin, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen2Backbone(nn.Module):
    """qwen2 backbone without lm head"""
    def __init__(self, vocab_size, hidden_size, num_hidden_layers,
                 num_attention_heads, num_kv_heads, intermediate_size,
                 max_position_embeddings):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2Block(hidden_size, num_attention_heads, num_kv_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        head_dim = hidden_size // num_attention_heads
        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                output_hidden_states=False):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        cos, sin = self.rotary_emb(x.size(1))
        cos = cos.to(x.dtype).to(x.device)
        sin = sin.to(x.dtype).to(x.device)

        all_hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(x)

        x = self.norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        return x, all_hidden_states


class Qwen2LMHead(nn.Module):
    """full qwen2 causal lm"""
    def __init__(self, vocab_size, hidden_size, num_hidden_layers,
                 num_attention_heads, num_kv_heads, intermediate_size,
                 max_position_embeddings):
        super().__init__()
        self.model = Qwen2Backbone(
            vocab_size, hidden_size, num_hidden_layers,
            num_attention_heads, num_kv_heads, intermediate_size,
            max_position_embeddings,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
        self.vocab_size = vocab_size

    def forward(self, input_ids=None, inputs_embeds=None, labels=None,
                attention_mask=None, output_hidden_states=False, **kwargs):
        hidden, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return type("Output", (), {
            "logits": logits,
            "loss": loss,
            "hidden_states": all_hidden_states,
        })()


# --- vaxseer time-dependent wrapper ---

class Qwen2TimeModel(nn.Module):
    """qwen2 with time-dependent rate + offset"""
    def __init__(self, config):
        super().__init__()
        num_kv_heads = max(config["num_attention_heads"] // 4, 1)
        intermediate_size = int(config["hidden_size"] * 8 / 3)
        intermediate_size = ((intermediate_size + 63) // 64) * 64

        self.rate_model = Qwen2LMHead(
            config["vocab_size"], config["hidden_size"],
            config["num_hidden_layers"], config["num_attention_heads"],
            num_kv_heads, intermediate_size,
            config["max_position_embeddings"],
        )

        self.transformer_offset = config.get("transformer_offset", False)
        if self.transformer_offset:
            self.offset_model = Qwen2LMHead(
                config["vocab_size"], config["hidden_size"],
                config["num_hidden_layers"], config["num_attention_heads"],
                num_kv_heads, intermediate_size,
                config["max_position_embeddings"],
            )
        else:
            self.offset_layer = nn.Linear(config["hidden_size"], config["vocab_size"])

        self.normalize_time_a = config.get("normalize_time_a", 1)
        self.normalize_time_b = config.get("normalize_time_b", 0)
        self.vocab_size = config["vocab_size"]
        self._config = config

    def forward(self, input_time, input_ids=None, labels=None,
                attention_mask=None, inputs_embeds=None,
                output_hidden_states=False, **kwargs):
        time = discretize_time(
            input_time, one_step=False,
            normalize_time_a=self.normalize_time_a,
            normalize_time_b=self.normalize_time_b,
            discrete=False,
        )
        beam_size = input_ids.size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)

        outputs = self.rate_model(
            input_ids=input_ids, labels=labels,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )

        rate = outputs.logits
        logits = rate * time.unsqueeze(-1).unsqueeze(-1)

        if self.transformer_offset:
            offset_out = self.offset_model(
                input_ids=input_ids, labels=labels,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            offset = offset_out.logits
        else:
            hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
            offset = self.offset_layer(hidden)

        logits = logits + offset
        outputs.logits = logits
        return outputs


# --- pytorch lightning wrapper ---

@register_model("qwen2_time")
class Qwen2TimeNew(pl.LightningModule):
    def __init__(self, config, alphabet, **kwargs) -> None:
        super().__init__()
        self.config = config

        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()

        vocab_size = len(alphabet) if kwargs.get("vocab_size") is None else kwargs.get("vocab_size")

        model_config = {
            "vocab_size": vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.hidden_size // 64,
            "max_position_embeddings": config.max_position_embeddings,
            "transformer_offset": getattr(config, "transformer_offset", False),
            "normalize_time_a": getattr(config, "normalize_time_a", 1),
            "normalize_time_b": getattr(config, "normalize_time_b", 0),
        }

        self.model = Qwen2TimeModel(model_config)

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Qwen2TimeNew total parameters: {total_params:,}")

    def initialize_model(self, pretrained_model_name_or_path: str):
        pass

    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            # estimate from trainer
            num_training_steps = self.trainer.estimated_stepping_batches
        if isinstance(num_warmup_steps, float):
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1, num_warmup_steps=0.1,
        )
        if self.config.scheduler == "none":
            return {"optimizer": optimizer}
        elif self.config.scheduler == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.scheduler == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None,
                             hparams_file=None, strict=True,
                             hf_pipeline_kwargs=None, args=None, **kwargs):
        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")
        # rebuild alphabet from saved hyperparameters
        from data.vocab import load_esm_alphabet
        alphabet = load_esm_alphabet(args.vocab, args.mol_type)
        model = cls(args, alphabet=alphabet)
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        model.config.resume_from_checkpoint = checkpoint_path
        model.config.pred_data_paths = getattr(args, "pred_data_paths", "")
        if args is not None:
            model.config.test_data_paths = args.test_data_paths
        for key in kwargs:
            setattr(model, key, kwargs[key])
        return model

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser.add_argument('--load_weights', action='store_true')
        parent_parser.add_argument('--num_hidden_layers', type=int, default=12)
        parent_parser.add_argument('--tau', type=float, default=1.0)
        parent_parser.add_argument('--hidden_size', type=int, default=768)
        parent_parser.add_argument('--model_name_or_path', type=str, default="gpt2")
        parent_parser.add_argument('--load_from_pretrain_checkpoint', type=str, default=None)
        parent_parser.add_argument('--normalize_time_a', type=int, default=1)
        parent_parser.add_argument('--normalize_time_b', type=int, default=0)
        parent_parser.add_argument('--add_location', action='store_true')
        parent_parser.add_argument('--add_lineage', action='store_true')
        parent_parser.add_argument('--weight_loss_by_count', type=str2bool, default="false")
        parent_parser.add_argument('--no_normalization_in_batch', action='store_true')
        parent_parser.add_argument('--zero_offset', action='store_true')
        parent_parser.add_argument('--offset_share_layer', type=int, default=-1)
        parent_parser.add_argument('--transformer_offset', action='store_true')
        parent_parser.add_argument('--second_order_rate', action='store_true')
        parent_parser.add_argument('--transformer_second_order_rate', action='store_true')
        parent_parser.add_argument('--output_token_losses', type=str2bool, default="false")
        parent_parser.add_argument('--do_sample', type=str2bool, default="false")
        parent_parser.add_argument('--temperature', type=float, default=1.0)
        parent_parser.add_argument('--num_beams', type=int, default=1)
        parent_parser.add_argument('--num_return_sequences', type=int, default=1)
        parent_parser.add_argument('--zero_time', action='store_true')
        parent_parser.add_argument('--set_time', type=float, default=None)
        parent_parser.add_argument('--ensemble', type=str2bool, default="false")
        parent_parser.add_argument('--average_over_time', type=str2bool, default="false")
        parent_parser.add_argument('--freeze_params_before_layer', type=int, default=0)
        parent_parser.add_argument('--weight_loss_by_time', type=str2bool, default="false")
        return parent_parser

    def nll_loss(self, lm_logits, labels, loss_weight=None, reduce=True):
        labels = labels.masked_fill(torch.eq(labels, self.alphabet.pad()), -100)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduce=False)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        if reduce:
            loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
            if loss_weight is not None:
                if not self.config.no_normalization_in_batch:
                    loss_weight = loss_weight / loss_weight.sum()
                loss = torch.sum(loss * loss_weight)
            else:
                loss = loss.mean()
        return loss

    def get_offset(self, batch, outputs=None):
        if getattr(self.config, "transformer_offset", False):
            offset_out = self.model.offset_model(
                input_ids=batch["input_ids"], labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            return offset_out.logits
        else:
            hidden = outputs.hidden_states[getattr(self.config, "offset_share_layer", -1)]
            return self.model.offset_layer(hidden)

    def get_unnorm_nll(self, rate_logits, labels, reduce=True):
        loss = -nn.NLLLoss(reduce=False)(
            rate_logits.view(-1, rate_logits.size(-1)), labels.view(-1)
        )
        loss = loss.view(labels.size())
        return loss.sum(-1) if reduce else loss

    def core(self, batch):
        x = self.model.rate_model.model.embed_tokens(batch["input_ids"])
        if self.config.add_location:
            x = x + self.location_embeddings(batch["location"]).unsqueeze(1)
        if getattr(self.config, "add_lineage", False):
            x = x + self.lineage_embeddings(batch["lineage"]).unsqueeze(1)
        outputs = self.model.rate_model(
            inputs_embeds=x, labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        return outputs

    def get_rate(self, outputs):
        return outputs.logits

    def testing_forward(self, batch, batch_idx, return_rate=False, return_offset=False):
        loss_weight = batch.get('freq', None)
        max_time, min_time = self.max_testing_time, self.min_testing_time
        input_times = torch.arange(min_time, max_time + 1).to(batch["input_ids"].device)
        time = discretize_time(
            input_times, one_step=False,
            normalize_time_a=self.config.normalize_time_a,
            normalize_time_b=self.config.normalize_time_b,
            discrete=False,
        )
        outputs = self.core(batch)
        rate = self.get_rate(outputs).unsqueeze(0)
        logits = rate * time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / getattr(self.config, "tau", 1.0)
        if not self.config.zero_offset:
            offset = self.get_offset(batch, outputs).unsqueeze(0)
            logits = logits + offset
        labels = batch["labels"]
        labels = labels.masked_fill(torch.eq(labels, self.alphabet.pad()), -100)
        repeat_labels = labels.unsqueeze(0).repeat(logits.size(0), 1, 1)
        loss = self.nll_loss(
            logits.view(-1, logits.size(2), logits.size(3)),
            repeat_labels.view(-1, repeat_labels.size(2)),
            loss_weight=loss_weight, reduce=False,
        )
        loss = loss.view(logits.size(0), -1)
        loss_dict = {}
        if return_rate:
            loss_dict["rate"] = self.get_unnorm_nll(rate.squeeze(0), labels)
        if return_offset and not self.config.zero_offset:
            loss_dict["offset"] = self.get_unnorm_nll(offset, labels)
        return loss, loss_dict

    def forward(self, batch, batch_idx, reduce=True, return_rate=False,
                return_offset=False, mode="train"):
        if getattr(self.config, "zero_time", False):
            batch["input_time"].fill_(0.0)
        if getattr(self.config, "set_time", None) is not None:
            batch["input_time"].fill_(self.config.set_time)

        logits = self.model(**batch).logits / self.config.temperature

        if self.config.weight_loss_by_count and batch.get('freq') is not None and batch.get('bin_size') is not None:
            loss_weight = batch['freq'] * batch['bin_size']
        elif not self.config.weight_loss_by_count and batch.get('freq') is not None:
            loss_weight = batch['freq']
        else:
            loss_weight = 1.0

        loss = self.nll_loss(logits, batch["labels"], loss_weight=loss_weight, reduce=reduce)
        return loss, {}

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, _ = self.forward(batch, batch_idx, reduce=False, mode="test")
        token_num = torch.sum(
            (batch["labels"][..., 1:].contiguous() != self.alphabet.pad())
            * (batch["labels"][..., 1:].contiguous() != self.alphabet.eos())
            * (batch["labels"][..., 1:].contiguous() != self.alphabet.bos()),
            dim=-1,
        )
        if "freq" in batch and "bin_size" in batch:
            weight = batch["freq"] * batch["bin_size"]
        else:
            weight = token_num.new_zeros(token_num.size(0)) + 1.0
        self.log("test_loss", loss.mean(), prog_bar=True)
        return loss, token_num, weight

    def test_epoch_end(self, outputs):
        losses, token_nums, weights = [], [], []
        if len(self.config.test_data_paths) == 1:
            outputs = [outputs]
        for dl_outputs in outputs:
            for output in dl_outputs:
                losses.append(output[0].sum(-1))
                token_nums.append(output[1])
                weights.append(output[2])
        losses = torch.cat(losses)
        token_nums = torch.cat(token_nums)
        weights = torch.cat(weights)
        ppl = torch.exp(torch.sum(losses * weights) / torch.sum(token_nums * weights))
        nll = torch.sum(weights * losses) / torch.sum(weights)
        self.log_dict({"perplexity": ppl, "nll": nll, "coverage": torch.exp(-losses).sum()})
        if self.config.output_token_losses:
            self.all_outputs = [x for dl in outputs for o in dl for x in o[0]]
        else:
            self.all_outputs = [
                {"prediction": l.item(), "token_num": t.item()}
                for l, t in zip(losses, token_nums)
            ]
        return ppl

    def output_testing_results(self, outputs, predict_dataset):
        predict_dataset = [item for sublist in predict_dataset for item in sublist]
        assert len(outputs) == len(predict_dataset)
        results = []
        for i, out in enumerate(outputs):
            if self.config.output_token_losses:
                d = {"prediction": " ".join([str(x.item()) for x in out])}
            else:
                d = out
            d["src_id"] = predict_dataset[i]["src_id"]
            d["src_time"] = predict_dataset[i]["src_time"]
            d["freq"] = predict_dataset[i]["freq"]
            results.append(d)
        return results

    def output_predicting_results(self, outputs, predict_dataset, *args, **kwargs):
        return [{"prediction": o["prediction"], "src_time": o["src_time"]} for o in outputs]
