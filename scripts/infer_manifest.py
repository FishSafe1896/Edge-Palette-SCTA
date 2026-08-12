from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from model.configuration import TransModule_Config  # noqa: E402
from model.framework import Encoder  # noqa: E402
from net import Decoder_MVGG, TransModule  # noqa: E402
from tools import Sample_Test_Net, content_style_transTo_pt, load_network_weights  # noqa: E402


def batched_rows(rows: list[dict[str, str]], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def build_network(use_style_adapter: bool = False, style_adapter_alpha: float = 0.1) -> Sample_Test_Net:
    config = TransModule_Config(
        nlayer=3,
        d_model=768,
        nhead=8,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    )
    encoder = Encoder(
        img_size=224,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        nhead=[3, 6, 12],
        strip_width=[2, 4, 7],
        drop_path_rate=0.0,
        patch_norm=True,
    )
    decoder = Decoder_MVGG(d_model=768, seq_input=True)
    trans_module = TransModule(
        config,
        use_style_adapter=use_style_adapter,
        style_adapter_alpha=style_adapter_alpha,
    )
    return Sample_Test_Net(encoder, decoder, trans_module)


def load_checkpoint(
    network: Sample_Test_Net,
    checkpoint_path: Path,
    device: torch.device,
    allow_partial_transmodule: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_network_weights(
        network,
        checkpoint,
        allow_partial_transmodule=allow_partial_transmodule,
    )


def set_ppeb_mode(network: Sample_Test_Net, mode: str) -> None:
    for module in network.modules():
        if hasattr(module, "set_mode"):
            module.set_mode(mode)


@torch.no_grad()
def infer_manifest(
    manifest_csv: Path,
    output_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    ppeb_mode: str,
    use_style_adapter: bool,
    style_adapter_alpha: float,
    skip_existing: bool = False,
    batch_size: int = 1,
) -> None:
    network = build_network(
        use_style_adapter=use_style_adapter,
        style_adapter_alpha=style_adapter_alpha,
    )
    load_checkpoint(
        network,
        checkpoint_path,
        device,
        allow_partial_transmodule=use_style_adapter,
    )
    set_ppeb_mode(network, ppeb_mode)
    network.to(device)
    network.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pending_rows = []
    for row in rows:
        output_path = output_dir / f'{row["pair_id"]}.jpg'
        if skip_existing and output_path.exists():
            continue
        pending_rows.append(row)

    progress = tqdm(total=len(pending_rows), desc="infer")
    for batch in batched_rows(pending_rows, batch_size):
        loaded = []
        for row in batch:
            content_path = resolve_project_path(row["content_path"])
            style_path = resolve_project_path(row["style_path"])
            i_c, i_s = content_style_transTo_pt(content_path, style_path)
            loaded.append((row, i_c, i_s))

        grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list] = {}
        for item in loaded:
            _, i_c, i_s = item
            grouped.setdefault((tuple(i_c.shape), tuple(i_s.shape)), []).append(item)

        for group in grouped.values():
            rows_group = [item[0] for item in group]
            content_batch = torch.cat([item[1] for item in group], dim=0).to(device)
            style_batch = torch.cat([item[2] for item in group], dim=0).to(device)
            outputs = network(content_batch, style_batch, arbitrary_input=True).cpu()
            for row, output in zip(rows_group, outputs):
                save_image(output, output_dir / f'{row["pair_id"]}.jpg')
        progress.update(len(batch))
    progress.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=Path, default=ROOT / "data" / "manifests" / "test_pairs.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_import_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ppeb_mode", type=str, default="full", choices=["off", "edge", "freq", "full"])
    parser.add_argument("--use_style_adapter", action="store_true")
    parser.add_argument("--style_adapter_alpha", type=float, default=0.1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    infer_manifest(
        manifest_csv=args.manifest_csv,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_import_path,
        device=device,
        ppeb_mode=args.ppeb_mode,
        use_style_adapter=args.use_style_adapter,
        style_adapter_alpha=args.style_adapter_alpha,
        skip_existing=args.skip_existing,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
