"""
Qwen3-Omni-30B-A3B-Instruct wrapper with the same interface as steering/qwen_omni.py (QwenOmni),
so steering/run_steps.py runs on it with `--model qwen3-omni`.

  Thinker  model.thinker.model.layers   48 layers, d = 2048, MoE (128 experts, 8 active)
  Talker   model.talker.model.layers    20 layers, d = 1024, MoE
  Bridge   two MLPs, not one linear map: talker.text_projection (Thinker token embeddings ->
           Talker, for text tokens) and talker.hidden_projection (Thinker layer-24 hidden
           states -> Talker, for audio tokens). Cross-stream steering (exp4a / exp4b) needs a
           linear bridge, so it is not offered for this model.

What differs from Qwen2.5-Omni, and how this file handles it:
  - generate() always returns (thinker_result, audio_or_None) -> normalised in generate().
  - The Talker skips the system turn: its prompt is the user turn (same tokens as the
    Thinker's) followed by a rebuilt assistant prefix. So a Thinker position p in the user
    turn is position p - user_turn_start in the Talker (anchor_position()).
  - The audio token id comes from the config (151675 in the released checkpoint, not the
    library default 151646).
  - Two input modes. "audio" (SpeechBreak's default): the spoken request, read at the last
    audio token, as for Qwen2.5-Omni. "text": the request's text (the manifest's
    spoken_text), read at the last prompt token, as for Kimi-Audio - this is the input
    VoiceBench's Qwen3-Omni agent gets, so its direction is the one to hand to VoiceBench.

Memory: the weights are 70.5 GB in bf16, so the model needs a whole 80 GB A100.
load_talker=False drops the Talker + Code2Wav (a few GB) for steps that only use the Thinker.
"""

import gc
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from steering.hooks import add_hooks

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
STREAMS = ("thinker", "talker")
INPUT_MODES = ("audio", "text")
OUTPUT_SR = 24000
DEFAULT_SPEAKER = "Ethan"

# Neutral system prompt. Unlike Qwen2.5-Omni, Qwen3-Omni speaks without a fixed system prompt;
# the Talker ignores the system turn either way.
SYSTEM_PROMPT = "You are a helpful assistant."


class _PrefillCaptured(Exception):
    """Raised from a capture hook to stop generation once the prefill activations are recorded."""


class Qwen3Omni:
    # experiments of steering/run_steps.py this model supports (no linear Thinker <-> Talker bridge)
    EXPERIMENTS = ("exp1_thinker", "exp2_talker", "exp3_both")

    def __init__(self, model_id: str = MODEL_ID, input_mode: str = "audio", load_talker: bool = True,
                 speaker: str = DEFAULT_SPEAKER):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        assert input_mode in INPUT_MODES, input_mode
        self.input_mode, self.speaker = input_mode, speaker
        # device_map="auto" rather than .to(): 70.5 GB of weights only just fit on one 80 GB GPU,
        # and accelerate places them without a second full copy in host memory
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()
        if not load_talker and self.model.has_talker:
            self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        self.device = self.model.thinker.device
        cfg = self.model.config
        self.audio_token_id = cfg.thinker_config.audio_token_id
        self.im_start_id, self.user_id = cfg.im_start_token_id, cfg.user_token_id

    # ------------------------------------------------------------------ #
    #  Architecture                                                        #
    # ------------------------------------------------------------------ #

    def blocks(self, stream: str) -> List[nn.Module]:
        """Transformer blocks of the 'thinker' or 'talker' residual stream."""
        if stream == "talker":
            self._need_talker()
            return list(self.model.talker.model.layers)
        return list(self.model.thinker.model.layers)

    @property
    def thinker_to_talker(self) -> torch.Tensor:
        raise NotImplementedError("Qwen3-Omni has no linear Thinker -> Talker projection "
                                  "(two MLPs instead); exp4a / exp4b are not supported")

    def _need_talker(self):
        if not self.model.has_talker:
            raise RuntimeError("the Talker was not loaded (load_talker=False)")

    # ------------------------------------------------------------------ #
    #  Inputs                                                              #
    # ------------------------------------------------------------------ #

    def _inputs(self, user_content: List[dict], audio=None) -> Dict[str, torch.Tensor]:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        kwargs = {"text": text, "return_tensors": "pt", "padding": True}
        if audio is not None:
            kwargs["audio"] = [audio]
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Model inputs for one spoken user turn (a wav file)."""
        import librosa
        wave = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate, mono=True)[0]
        return self._inputs([{"type": "audio", "audio": audio_path}], audio=wave)

    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Model inputs for one written user turn."""
        return self._inputs([{"type": "text", "text": text}])

    def process_item(self, item: dict, repo_root: str = "") -> Dict[str, torch.Tensor]:
        """Inputs for a dataset item in this wrapper's input mode."""
        if self.input_mode == "text":
            return self.process_text(item.get("spoken_text") or item["instruction"])
        import os
        return self.process_audio(os.path.join(repo_root, item["audio_path"]))

    def anchor_position(self, inputs: Dict[str, torch.Tensor], stream: str = "thinker") -> int:
        """
        Where activations are read. Audio mode: the last <|AUDIO|> token (as for Qwen2.5-Omni).
        Text mode: the last prompt token (as for Kimi-Audio). For the Talker the index is shifted by
        the start of the user turn, because the Talker's prompt leaves out the system turn.
        """
        ids = inputs["input_ids"][0]
        if self.input_mode == "audio":
            pos = int((ids == self.audio_token_id).nonzero()[-1].item())
        else:
            pos = ids.shape[0] - 1
        if stream == "talker":
            if self.input_mode == "text":
                raise ValueError("text mode reads the last prompt token, which the Talker doesn't see "
                                 "(it rebuilds the assistant prefix); use audio mode for Talker activations")
            starts = (ids == self.im_start_id).nonzero().flatten().tolist()
            user_start = next(s for s in starts if s + 1 < len(ids) and ids[s + 1] == self.user_id)
            pos -= user_start
        return pos

    def last_audio_position(self, inputs: Dict[str, torch.Tensor]) -> int:
        """QwenOmni-compatible name for the Thinker anchor."""
        return self.anchor_position(inputs, "thinker")

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, inputs: Dict[str, torch.Tensor], max_new_tokens: int = 128,
                 talker_max_new_tokens: int = 1500, return_audio: Optional[bool] = None) -> Dict[str, Any]:
        """
        Reply: {"audio": waveform tensor at 24 kHz (empty when not spoken), "text": [Thinker text]}.
        max_new_tokens bounds the Thinker's text; talker_max_new_tokens bounds the Talker's codec
        tokens. return_audio defaults to whether the Talker is loaded.
        """
        speak = self.model.has_talker if return_audio is None else return_audio
        kwargs = dict(thinker_max_new_tokens=max_new_tokens, use_audio_in_video=False, return_audio=speak)
        if speak:
            kwargs.update(talker_max_new_tokens=talker_max_new_tokens, speaker=self.speaker)
        with torch.inference_mode():
            try:
                result, audio = self.model.generate(**inputs, **kwargs)
            except RuntimeError as e:
                # as for Qwen2.5-Omni: a Talker that emits no codec tokens (e.g. under strong
                # steering) makes the vocoder fail on an empty sequence - a silent reply, not an error
                if not speak or "0 elements" not in str(e):
                    raise
                kwargs.update(return_audio=False)
                kwargs.pop("talker_max_new_tokens", None), kwargs.pop("speaker", None)
                result, audio = self.model.generate(**inputs, **kwargs)
                audio = None
        seqs = result.sequences if hasattr(result, "sequences") else result
        new_ids = seqs[:, inputs["input_ids"].shape[-1]:]
        audio = audio.reshape(-1) if audio is not None else torch.zeros(0)
        return {"audio": audio, "text": self.processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)}

    def _run_prefill(self, inputs: Dict[str, torch.Tensor], stream: str):
        """The cheapest generate() call that runs the stream's prefill (aborted from a hook)."""
        if stream == "thinker":
            self.model.generate(**inputs, thinker_max_new_tokens=1, use_audio_in_video=False, return_audio=False)
        else:  # the Talker's prefill includes the first Thinker reply token, so >= 2 Thinker tokens
            self._need_talker()
            self.model.generate(**inputs, thinker_max_new_tokens=2, thinker_do_sample=False,
                                talker_max_new_tokens=2, use_audio_in_video=False, return_audio=True,
                                speaker=self.speaker)

    @torch.inference_mode()
    def talker_first_logprobs(self, inputs: Dict[str, torch.Tensor], hooks) -> torch.Tensor:
        """
        Log-probs of the Talker's first codec token [V] (step 3's Talker score). The Thinker runs
        greedily for 2 tokens, the Talker runs its prefill, and generation stops at the codec head.
        """
        self._need_talker()
        store = {}

        def capture(module, args, output):
            store["lp"] = F.log_softmax(output[0, -1].float(), dim=-1)
            raise _PrefillCaptured()

        with add_hooks(module_forward_pre_hooks=hooks,
                       module_forward_hooks=[(self.model.talker.codec_head, capture)]):
            try:
                self._run_prefill(inputs, "talker")
            except _PrefillCaptured:
                pass
        return store["lp"]

    # ------------------------------------------------------------------ #
    #  Activations                                                         #
    # ------------------------------------------------------------------ #

    def extract_activations(self, items: List[Union[str, dict]], stream: str, repo_root: str = "") -> torch.Tensor:
        """
        Residual stream (block input) at the anchor position of each prompt, from the stream's
        prefill pass, for every layer: [n, n_layers, d] (float32, cpu). Items are dataset items
        (or wav paths, in audio mode). Generation stops once the last layer has fired.
        """
        blocks = self.blocks(stream)
        last = len(blocks) - 1
        state = {"pos": None, "done": set(), "acts": {}}

        def make_hook(layer):
            def hook_fn(module, args):
                if layer in state["done"]:  # only the prefill, not the decode steps
                    return
                hidden = args[0] if args else None
                if hidden is None:
                    raise RuntimeError(f"{stream} layer {layer}: no positional hidden-state input to hook")
                state["acts"][layer] = hidden[0, state["pos"], :].detach().float().cpu()
                state["done"].add(layer)
                if layer == last:
                    raise _PrefillCaptured()
            return hook_fn

        hooks = [(b, make_hook(l)) for l, b in enumerate(blocks)]
        out = []
        for it in items:
            inputs = self.process_audio(it) if isinstance(it, str) else self.process_item(it, repo_root)
            state.update(pos=self.anchor_position(inputs, stream), done=set(), acts={})
            with torch.inference_mode(), add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                try:
                    self._run_prefill(inputs, stream)
                except _PrefillCaptured:
                    pass
            if len(state["done"]) != len(blocks):
                missing = sorted(set(range(len(blocks))) - state["done"])
                raise RuntimeError(f"{it if isinstance(it, str) else it.get('audio_path')}: "
                                   f"{stream} layers {missing} never ran")
            out.append(torch.stack([state["acts"][l] for l in range(len(blocks))]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.stack(out)

    def unload(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
