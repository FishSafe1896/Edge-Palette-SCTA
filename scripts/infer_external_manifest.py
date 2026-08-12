from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "src" / "baselines"


@contextmanager
def import_from(repo: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def infer_stytr2(rows: list[dict[str, str]], output_dir: Path, device: torch.device) -> None:
    repo = BASELINES / "StyTR-2"
    with import_from(repo):
        import models.StyTR as StyTR  # type: ignore
        import models.transformer as transformer  # type: ignore

        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load(repo / "experiments" / "vgg_normalised.pth", map_location=device))
        vgg = nn.Sequential(*list(vgg.children())[:44])

        decoder = StyTR.decoder
        decoder.load_state_dict(torch.load(repo / "experiments" / "decoder_iter_160000.pth", map_location=device))

        trans = transformer.Transformer()
        trans.load_state_dict(torch.load(repo / "experiments" / "transformer_iter_160000.pth", map_location=device))

        embedding = StyTR.PatchEmbed()
        embedding.load_state_dict(torch.load(repo / "experiments" / "embedding_iter_160000.pth", map_location=device))

        class Args:
            position_embedding = "sine"
            hidden_dim = 512

        network = StyTR.StyTrans(vgg, decoder, embedding, trans, Args())
        network.eval().to(device)

        image_tf = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])

        for row in tqdm(rows, desc="StyTR-2"):
            content = image_tf(Image.open(resolve_path(row["content_path"])).convert("RGB")).unsqueeze(0).to(device)
            style = image_tf(Image.open(resolve_path(row["style_path"])).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                output = network(content, style)
                if isinstance(output, tuple):
                    output = output[0]
            save_image(output.cpu(), output_dir / f'{row["pair_id"]}.jpg')


def infer_s2wat(rows: list[dict[str, str]], output_dir: Path, device: torch.device) -> None:
    repo = BASELINES / "S2WAT"
    with import_from(repo):
        from model.configuration import TransModule_Config  # type: ignore
        from model.s2wat import S2WAT  # type: ignore
        from net import Decoder_MVGG, TransModule  # type: ignore
        from tools import Sample_Test_Net, content_style_transTo_pt  # type: ignore

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
        encoder = S2WAT(
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
        trans_module = TransModule(config)
        network = Sample_Test_Net(encoder, decoder, trans_module)

        checkpoint = torch.load(
            repo / "pre_trained_models" / "checkpoint" / "checkpoint_40000_epoch.pkl",
            map_location=device,
            weights_only=False,
        )
        network.encoder.load_state_dict(checkpoint["encoder"])
        network.decoder.load_state_dict(checkpoint["decoder"])
        network.transModule.load_state_dict(checkpoint["transModule"])
        network.eval().to(device)

        for row in tqdm(rows, desc="S2WAT"):
            content_path = resolve_path(row["content_path"])
            style_path = resolve_path(row["style_path"])
            with torch.no_grad():
                content, style = content_style_transTo_pt(content_path, style_path)
                output = network(content.to(device), style.to(device), arbitrary_input=True)
            save_image(output.cpu(), output_dir / f'{row["pair_id"]}.jpg')


def infer_cast(rows: list[dict[str, str]], output_dir: Path, device: torch.device) -> None:
    repo = BASELINES / "CAST_pytorch"
    with import_from(repo):
        from data.base_dataset import get_transform  # type: ignore
        from models import create_model  # type: ignore
        from options.test_options import TestOptions  # type: ignore
        from util import util  # type: ignore

        old_argv = sys.argv
        sys.argv = [
            "test.py",
            "--dataroot",
            "placeholder",
            "--name",
            "CAST_model",
            "--checkpoints_dir",
            "checkpoints",
            "--model",
            "cast",
            "--phase",
            "test",
            "--epoch",
            "latest",
            "--eval",
            "--num_test",
            str(len(rows)),
            "--gpu_ids",
            "0" if device.type == "cuda" else "-1",
        ]
        try:
            opt = TestOptions().parse()
        finally:
            sys.argv = old_argv

        opt.num_threads = 0
        opt.batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.display_id = -1
        model = create_model(opt)
        model.setup(opt)
        model.parallelize()
        model.eval()
        image_tf = get_transform(opt)

        for row in tqdm(rows, desc="CAST_pytorch"):
            content = image_tf(Image.open(resolve_path(row["content_path"])).convert("RGB")).unsqueeze(0)
            style = image_tf(Image.open(resolve_path(row["style_path"])).convert("RGB")).unsqueeze(0)
            model.set_input({
                "A": content,
                "B": style,
                "A_paths": [str(resolve_path(row["content_path"]))],
                "B_paths": [str(resolve_path(row["style_path"]))],
            })
            with torch.no_grad():
                model.test()
            image = util.tensor2im(model.fake_B)
            util.save_image(image, output_dir / f'{row["pair_id"]}.jpg')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["StyTR-2", "S2WAT", "CAST_pytorch"])
    parser.add_argument("--manifest_csv", type=Path, default=ROOT / "data" / "manifests" / "test_pairs.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_csv = args.manifest_csv if args.manifest_csv.is_absolute() else ROOT / args.manifest_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = read_manifest(manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "StyTR-2":
        infer_stytr2(rows, output_dir, device)
    elif args.method == "S2WAT":
        infer_s2wat(rows, output_dir, device)
    elif args.method == "CAST_pytorch":
        infer_cast(rows, output_dir, device)
    else:
        raise ValueError(args.method)


if __name__ == "__main__":
    main()
