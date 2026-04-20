import argparse
import datetime
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer")))

from eval.sample import init_tokenizer, encode_audio_to_tokens
from eval.eval_utils import run_postprocessing
from utils.load_model import load_model_from_checkpoint
import core.samplers.sampling as sampling


def gap_to_tokens(gap_ms):
    return max(1, round(gap_ms * 0.075))


def create_masked_wavs(input_dir, masked_dir, orig_dir, gap_len_ms, sample_rate=24000):
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    gap_sec = gap_len_ms / 1000.0

    for fname in os.listdir(input_dir):
        if not fname.endswith(".wav"):
            continue

        signal, sr_file = sf.read(os.path.join(input_dir, fname))

        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if sr_file != sample_rate:
            signal = resample(signal, int(len(signal) * sample_rate / sr_file))

        sf.write(os.path.join(orig_dir, fname), signal, sample_rate, subtype="PCM_16")

        duration_sec = len(signal) / sample_rate
        gap_start = max(0.0, (duration_sec / 2) - (gap_sec / 2))
        start_sample = int(round(gap_start * sample_rate))
        end_sample = min(len(signal), start_sample + int(round(gap_sec * sample_rate)))

        masked = signal.copy()
        masked[start_sample:end_sample] = 0.0
        sf.write(os.path.join(masked_dir, fname), masked, sample_rate, subtype="PCM_16")


def run_inference(model_path, gap_len_ms, wav_dir, output_dir, num_samples, steps,
                  token_fill_length, wavtokenizer_config, wavtokenizer_ckpt, max_files, device):
    os.makedirs(output_dir, exist_ok=True)

    tokens_per_sec = 75
    gap_sec = gap_len_ms / 1000.0

    print(f"Loading model from {model_path}...")
    model, graph, noise = load_model_from_checkpoint(model_path, device)
    model.eval()

    print("Loading WavTokenizer...")
    tokenizer = init_tokenizer(wavtokenizer_config, wavtokenizer_ckpt, device)

    files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")][:max_files]
    print(f"Running inference on {len(files)} file(s) with {steps} steps...")

    for filename in tqdm(files):
        file_path = os.path.join(wav_dir, filename)
        token = encode_audio_to_tokens(file_path, tokenizer, device)
        token = torch.squeeze(token).view(1, 1, -1).to(device)

        seq_len = token.shape[-1]
        duration_sec = seq_len / tokens_per_sec
        start_token = int(((duration_sec / 2) - (gap_sec / 2)) * tokens_per_sec)
        end_token = start_token + token_fill_length

        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[start_token:end_token] = True

        def proj_fun(x, _mask=mask, _token=token):
            x[:, ~_mask] = _token[0, 0, ~_mask].unsqueeze(0).repeat(x.size(0), 1)
            return x

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (num_samples, seq_len),
            "analytic", steps, device=device, proj_fun=proj_fun,
        )

        with torch.no_grad():
            sampled = sampling_fn(model)
        filled_tokens = proj_fun(sampled)[None]

        try:
            features = tokenizer.codes_to_features(filled_tokens)
        except Exception:
            features = tokenizer.codes_to_features(filled_tokens.cpu())

        bandwidth_id = torch.tensor([0], device=device)
        audio_out = tokenizer.decode(features, bandwidth_id=bandwidth_id)

        stem = os.path.splitext(filename)[0]
        for i in range(num_samples):
            out_path = os.path.join(output_dir, f"{stem}_gen{i}.wav")
            audio = torch.clamp(audio_out[i][None], -1.0, 1.0).to(torch.float32).cpu()
            torchaudio.save(out_path, audio, sample_rate=24000, encoding="PCM_S", bits_per_sample=16)


def main():
    parser = argparse.ArgumentParser(description="AIDD audio inpainting inference")
    parser.add_argument("--input_dir", required=True, help="Directory of original .wav files")
    parser.add_argument("--output_dir", required=True,
                        default=f"results/{datetime.datetime.now().strftime('%Y.%m.%d_%H%M%S')}",
                        help="Root directory for all outputs")
    parser.add_argument("--model_path", required=True,
                        help="Path to checkpoint.pth (config.yaml must be in the same directory)")
    parser.add_argument("--wavtokenizer_config",
                        default="./tokenizer/configs/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    parser.add_argument("--wavtokenizer_ckpt",
                        default="./tokenizer/models/wavtokenizer_medium_music_audio_320_24k.ckpt")
    parser.add_argument("--gaps", type=int, default=350, help="Gap length in ms")
    parser.add_argument("--steps", type=int, default=1024, help="Diffusion sampling steps")
    parser.add_argument("--samples", type=int, default=1, help="Inpaintings per file")
    parser.add_argument("--max_files", type=int, default=500)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fade_len", type=int, default=100, help="Crossfade length in samples")
    parser.add_argument("--stitch", action="store_true",
                        help="Run postprocessing to stitch generated audio into originals")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    masked_dir = os.path.join(args.output_dir, "masked_wavs")
    orig_dir = os.path.join(args.output_dir, "orig_wavs")
    gen_dir = os.path.join(args.output_dir, "generated_wavs")
    stitched_dir = os.path.join(args.output_dir, "stitched")

    print(f"\n=== Creating masked audio (gap={args.gaps}ms) ===")
    create_masked_wavs(args.input_dir, masked_dir, orig_dir, args.gaps)

    token_fill_length = gap_to_tokens(args.gaps)

    print(f"\n=== Running inference ===")
    run_inference(
        model_path=args.model_path,
        gap_len_ms=args.gaps,
        wav_dir=masked_dir,
        output_dir=gen_dir,
        num_samples=args.samples,
        steps=args.steps,
        token_fill_length=token_fill_length,
        wavtokenizer_config=args.wavtokenizer_config,
        wavtokenizer_ckpt=args.wavtokenizer_ckpt,
        max_files=args.max_files,
        device=device,
    )

    if args.stitch:
        print(f"\n=== Stitching generated audio ===")
        run_postprocessing(
            gap_len_ms=args.gaps,
            masked_dir=masked_dir,
            gen_dir=gen_dir,
            output_dir=stitched_dir,
            token_fill_length=token_fill_length,
            fade_len=args.fade_len,
        )

    print(f"\n=== Done ===")
    print(f"  Masked audio : {masked_dir}")
    print(f"  Generated    : {gen_dir}")
    if args.stitch:
        print(f"  Stitched     : {stitched_dir}")


if __name__ == "__main__":
    main()
