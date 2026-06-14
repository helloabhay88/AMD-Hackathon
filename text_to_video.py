import os
import sys

try:
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video
    import torch
except ImportError:
    print("\n[!] Required packages not found. Auto-installing dependencies...")
    import subprocess
    try:
        # CogVideoX requires diffusers, transformers, and accelerate
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "diffusers>=0.30.0", "transformers", "accelerate", "imageio[ffmpeg]"
        ])
        print("[✔] Dependencies installed successfully! Loading libraries...\n")
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
        import torch
    except Exception as e:
        print(f"\n[❌] Failed to auto-install packages: {str(e)}")
        print("Please run manually: pip install diffusers transformers accelerate imageio[ffmpeg]\n")
        sys.exit(1)

def main():
    # Model configuration (CogVideoX 2B is fast and fits easily in VRAM)
    model_id = "THUDM/CogVideoX-2b"
    
    print("\n" + "="*50)
    print("      AMD GPU LOCAL TEXT-TO-VIDEO GENERATOR")
    print("="*50)
    
    prompt = input("\nEnter prompt to generate video (e.g. 'A close-up of a panda eating bamboo in a forest, cinematic'): \n> ")
    if not prompt.strip():
        prompt = "A futuristic city with flying cars zooming between skyscrapers at sunset, 3d render, hyperrealistic"
        print(f"Using default prompt: '{prompt}'")
        
    output_filename = "generated_video.mp4"
    
    # Detect GPU (ROCm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] CUDA (ROCm) status: {torch.cuda.is_available()} - Running on: {device}")
    
    print(f"[*] Loading model: {model_id}...")
    try:
        # Load the pipeline in bfloat16 (required for CogVideoX VAE stability)
        pipe = CogVideoXPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
            
        print("\n[*] Generating video frames. This will take a moment on your AMD GPU...")
        
        # Run inference (default generates 6 seconds of video at 8 fps)
        video_frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,  # 50 steps is standard for quality
            guidance_scale=6.0,
            generator=torch.manual_seed(42)
        ).frames[0]
        
        # Save output video
        export_to_video(video_frames, output_filename, fps=8)
        print(f"\n[✔] Video successfully generated and saved to: '{os.path.abspath(output_filename)}'\n")
        
    except Exception as e:
        print(f"\n[❌] An error occurred during model loading/execution: {str(e)}")
        print("Make sure you have sufficient hardware resources and permissions.\n")

if __name__ == "__main__":
    main()
