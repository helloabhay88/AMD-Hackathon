import os
import sys

try:
    from diffusers import DiffusionPipeline
    import torch
except ImportError:
    print("\n[!] Required image generation packages not found. Auto-installing dependencies...")
    import subprocess
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "diffusers", "accelerate", "invisible-watermark>=0.2.0"
        ])
        print("[✔] Dependencies installed successfully! Loading libraries...\n")
        from diffusers import DiffusionPipeline
        import torch
    except Exception as e:
        print(f"\n[❌] Failed to auto-install packages: {str(e)}")
        print("Please run manually: pip install diffusers accelerate invisible-watermark>=0.2.0\n")
        sys.exit(1)

def main():
    # Model configuration (Stable Diffusion XL Base 1.0)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    print("\n" + "="*50)
    print("      AMD GPU LOCAL IMAGE GENERATOR (SDXL)")
    print("="*50)
    
    # 1. Prompt Input
    print("\n[Tip] Be descriptive! E.g. 'A professional studio portrait of a man, sharp focus, 85mm lens, photorealistic'")
    prompt = input("Enter positive prompt: \n> ")
    if not prompt.strip():
        prompt = "A high-tech digital laboratory with AMD Instinct MI300X servers glowing in neon blue, hyperrealistic, 8k resolution"
        print(f"Using default prompt: '{prompt}'")
        
    # 2. Negative Prompt Input
    default_negative = "blurry, bad anatomy, deformed, extra limbs, low quality, disfigured, cartoon, drawing, painting, 3d render"
    print(f"\n[Default Negative] '{default_negative}'")
    negative_prompt = input("Enter custom negative prompt (or press Enter to use default): \n> ")
    if not negative_prompt.strip():
        negative_prompt = default_negative
        
    output_filename = "generated_image.png"
    
    # Detect GPU (ROCm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] CUDA (ROCm) status: {torch.cuda.is_available()} - Running on: {device}")
    
    # Loading pipeline
    print(f"[*] Loading model: {model_id}...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            use_safetensors=True, 
            variant="fp16" if device == "cuda" else None
        )
        pipe = pipe.to(device)
        
        # Optimize memory usage
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
                
        print("\n[*] Generating image. Please wait...")
        print("    - Steps: 35 (Higher quality)")
        print("    - Guidance Scale: 7.5")
        
        # Run inference
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,
            guidance_scale=7.5
        ).images[0]
        
        # Save output image
        image.save(output_filename)
        print(f"\n[✔] Image successfully generated and saved to: '{os.path.abspath(output_filename)}'\n")
        
    except Exception as e:
        print(f"\n[❌] An error occurred during model loading/execution: {str(e)}")
        print("Make sure you have downloaded the weights successfully and have sufficient permissions.\n")

if __name__ == "__main__":
    main()
