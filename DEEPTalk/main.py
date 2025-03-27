# app/main.py
import os
import json
import time
import glob
import tempfile
import librosa
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from DEEPTalk.settings import settings

from DEEPTalk.models import DEMOTE_VQ
from DEEPTalk.utils.extra import seed_everything, compare_checkpoint_model

from DEE.get_DEE import get_DEE_from_json
from FER.get_model import init_affectnet_feature_extractor

app = FastAPI(title="DeepTalk Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {"message": "DeepTalk Server is Running!"}


class Args(BaseModel):
    DEEPTalk_model_path: str = os.path.join(
        settings.BASE_DIR, "DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth"
    )
    DEEPTalk_config_path: str = os.path.join(
        settings.BASE_DIR, "DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth"
    )
    DEE_ckpt_path: str = os.path.join(settings.BASE_DIR, "DEE/checkpoint/DEE.pt")
    audio_path: str = None
    use_sampling: bool = False
    control_logvar: float = None
    tau: float = 0.0001


DEMOTE_model_path = os.path.join(
    settings.BASE_DIR, "DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth"
)
DEMOTE_config_path = f"{os.path.dirname(DEMOTE_model_path)}/config.json"
with open(DEMOTE_config_path) as f:
    DEMOTE_config = json.load(f)

training_ids = [
    "M003",
    "M005",
    "M007",
    "M009",
    "M011",
    "M012",
    "M013",
    "M019",
    "M022",
    "M023",
    "M024",
    "M025",
    "M026",
    "M027",
    "M028",
    "M029",
    "M030",
    "M031",
    "W009",
    "W011",
    "W014",
    "W015",
    "W016",
    "W018",
    "W019",
    "W021",
    "W023",
    "W024",
    "W025",
    "W026",
    "W028",
    "W029",
]  # 32 ids
MEAD_ACTOR_DICT = {k: i for i, k in enumerate(training_ids)}


def pad_audio_to_match_quantfactor(audio_samples, fps=30, quant_factor=3):
    """padidng audio samples to be divisible by quant factor
    (NOTE) quant factor means the latents must be divisible by 2^(quant_factor)
           for inferno's EMOTE checkpoint, the quant_factor is 3 and fps is 25
    Args:
        audio_samples (torch tensor or numpy array): audio samples from raw wav files
        fps (int, optional): fps of the face parameters. Defaults to 30.
        quant_factor (int, optional): squaushing latent variables by 2^(quant_factor) Defaults to 8.
    """
    if isinstance(audio_samples, np.ndarray):
        audio_samples = torch.tensor(audio_samples, dtype=torch.float32)

    audio_len = audio_samples.shape[0]
    latent_len = int(audio_len / 16000 * fps)  # to target fps
    target_len = latent_len + (
        2**quant_factor - (latent_len % (2**quant_factor))
    )  # make sure the length is divisible by quant factor
    target_len = int(target_len / fps * 16000)  # to audio sample rate

    padded_audio_samples = torch.nn.functional.pad(
        audio_samples, (0, target_len - len(audio_samples))
    )
    if isinstance(audio_samples, np.ndarray):
        padded_audio_samples = padded_audio_samples.numpy()
    return padded_audio_samples


@app.on_event("startup")
async def load_models():
    """Load models during application startup."""
    global model, affectnet_feature_extractor, device, DEMOTE_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seed_everything(42)

    # Load FLINT configuration
    FLINT_config_path = DEMOTE_config["motionprior_config"]["config_path"]
    with open(FLINT_config_path, "r") as f:
        FLINT_config = json.load(f)
    FLINT_ckpt = DEMOTE_config["motionprior_config"]["checkpoint_path"]

    # Load DEE model
    DEE_config_path = glob.glob(f"{os.path.dirname(Args().DEE_ckpt_path)}/*.json")[0]
    print(f"DEE config loaded: {DEE_config_path}")
    DEE_model, DEE_config = get_DEE_from_json(DEE_config_path)
    DEE_checkpoint = torch.load(Args().DEE_ckpt_path, map_location="cpu")
    DEE_model.load_state_dict(DEE_checkpoint)
    DEE_model.eval()
    compare_checkpoint_model(DEE_checkpoint, DEE_model.to("cpu"))

    # Load AffectNet feature extractor if available
    affectnet_feature_extractor = None
    if DEE_config.affectnet_model_path:
        model_path = DEE_config.affectnet_model_path
        config_path = os.path.dirname(model_path) + "/config.yaml"
        _, affectnet_feature_extractor = init_affectnet_feature_extractor(
            config_path, model_path
        )
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)

    # Load DEMOTE model
    global model
    model = DEMOTE_VQ.DEMOTE_VQVAE_condition(
        DEMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE_model, load_motion_prior=False
    )
    DEMOTE_ckpt = torch.load(Args().DEEPTalk_config_path, map_location="cpu")
    model.load_state_dict(DEMOTE_ckpt)
    model.eval()
    model.to(device)
    print("Models loaded successfully.")

@app.router.post("/audio/flame-params")
async def get_flame_params(audio_file: UploadFile = File(...)):
    print(f"Received file: {audio_file.filename}")
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio_file.read())
        audio_path = temp_file.name

    args = Args(audio_path=audio_path)

    audio_processing_start_time = time.time()
    print("Processing audio...")

    # Load audio data
    audio_path = args.audio_path
    if audio_path.endswith(".wav"):
        wavdata, sampling_rate = librosa.load(audio_path, sr=16000)
    elif audio_path.endswith(".npy"):
        wavdata = np.load(audio_path)
    else:
        raise ValueError("Audio file must be either .wav or .npy")

    # Prepare output directories
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    param_save_dir = os.path.join(save_dir, "params")
    os.makedirs(param_save_dir, exist_ok=True)
    file_name = os.path.basename(audio_path).split(".")[0]
    output_name = f"{file_name}"
    out_param_path = os.path.join(param_save_dir, f"{output_name}.npy")

    if not os.path.exists(out_param_path):
        audio = torch.tensor(wavdata, dtype=torch.float32)
        audio = pad_audio_to_match_quantfactor(
            audio,
            fps=DEMOTE_config["audio_config"]["target_fps"],
            quant_factor=DEMOTE_config["sequence_decoder_config"]["quant_factor"],
        )

        emotion = 0  # Random number as DEMOTE does not use this
        intensity = 0  # Also a random number
        id = "M003"
        actor_id = MEAD_ACTOR_DICT[id]
        n_emotions = DEMOTE_config["sequence_decoder_config"]["style_embedding"][
            "n_expression"
        ]
        n_intensities = DEMOTE_config["sequence_decoder_config"]["style_embedding"][
            "n_intensities"
        ]
        n_identities = DEMOTE_config["sequence_decoder_config"]["style_embedding"][
            "n_identities"
        ]
        condition_size = n_emotions + n_intensities + n_identities
        input_style = torch.eye(condition_size)[
            [
                emotion,  # Emotion one hot
                n_emotions + intensity,  # Intensity one hot
                n_emotions + n_intensities + actor_id,
            ]
        ]  # Actor ID one hot

        input_style = torch.sum(input_style, dim=0).unsqueeze(0)  # (1,60)
        inputs = []

        print(f"Audio shape: {audio.shape}")
        audio = audio.unsqueeze(0)  # (1, audio_len)
        audio = audio.to(device)
        input_style = input_style.to(device)
        print(f"Audio: {audio.shape}")

        with torch.no_grad():
            output = model(
                audio,
                input_style,
                sample=args.use_sampling,
                control_logvar=args.control_logvar,
                tau=args.tau,
            )

        print(f"Output: {output.shape}")

        # Batch, Frame_num, Params
        np.save(out_param_path, output.squeeze(0).detach().cpu().numpy())
        print(f"Saved in {out_param_path}")
    else:
        output = torch.tensor(np.load(out_param_path)).unsqueeze(0).to(device)
    audio_processing_end_time = time.time()
    print(f"Audio processing time: {audio_processing_end_time - audio_processing_start_time}")

    print("Getting flame model and calculating vertices...")
    expression_params = output[:, :, :50]
    jaw_pose = output[:, :, 50:53]
    print(f"Expression params shape: {expression_params.shape}")
    print(f"Jaw pose shape: {jaw_pose.shape}")

    animation_params = []
    for i in range(len(expression_params[0])):
        animation_params.append(
            {
                "time": i / 25,
                "parameters": {
                    "jaw": jaw_pose[0][i].tolist(),
                    "neck": [0.0, 0.0, 0.0],
                    "eyes": [0.0, 0.0, 0.0],
                    "expr": expression_params[0][i].tolist(),
                },
            }
        )

    # Return animation_params as API response
    return animation_params


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
