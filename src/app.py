import os
import argparse
import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from src.dataset import PatchCamelyonDataset, get_transforms
from src.model import get_resnet18
from src.utils import normalize_stain_macenko

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_app_model(model_path):
    """
    Loads the ResNet-18 model from the specified checkpoint.
    """
    model = get_resnet18(pretrained=False, num_classes=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model checkpoint from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Running dashboard with un-trained/random weights.")
    
    model = model.to(device)
    model.eval()
    return model

def create_example_images(mock_mode, test_x=None, test_y=None):
    """
    Prepares sample slide images from the Test set on disk to be used as clickable Gradio examples.
    """
    samples_dir = "data/samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    examples_list = []
    indices_labels = [(42, "sample_healthy"), (120, "sample_tumour"), (250, "sample_borderline")]
    
    # Configure dataset
    _, val_test_transform = get_transforms()
    if mock_mode or not test_x or not os.path.exists(test_x):
        print("Using mock dataset to generate Gradio app example images...")
        dataset = PatchCamelyonDataset(split='test', num_mock_samples=500, transform=val_test_transform)
    else:
        print(f"Using real test dataset from {test_x} to generate Gradio app example images...")
        dataset = PatchCamelyonDataset(test_x, test_y, transform=val_test_transform)
        
    for idx, name in indices_labels:
        path = os.path.join(samples_dir, f"{name}.png")
        if dataset.is_mock:
            raw_arr = dataset.images[idx]
        else:
            raw_arr = dataset.x_data[idx]
            
        Image.fromarray(raw_arr).save(path)
        examples_list.append([path, True])
        
    return examples_list

def run_diagnostics(img_input, apply_norm, model):
    if img_input is None:
        return (
            "<div style='text-align: center; color: #94a3b8; padding: 20px;'>⚠️ Please upload an image first.</div>",
            None
        )
        
    # Convert input to RGB PIL Image
    raw_img = Image.fromarray(img_input.astype('uint8'), 'RGB')
    
    # 1. Stain Normalization
    if apply_norm:
        img_np = np.array(raw_img.resize((96, 96)))
        try:
            normalized_np = normalize_stain_macenko(img_np)
            processed_img = Image.fromarray(normalized_np)
        except Exception as e:
            print(f"Macenko Normalization failed: {e}. Falling back to standard image resizing.")
            processed_img = raw_img.resize((96, 96))
            normalized_np = img_np
    else:
        processed_img = raw_img.resize((96, 96))
        normalized_np = np.array(processed_img)
        
    # 2. PyTorch Evaluation
    inference_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = inference_transform(processed_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prob_healthy = probabilities[0][0].item()
        prob_tumour = probabilities[0][1].item()
        
    # 3. Render Custom HTML Verdict and glowing progress bars
    is_tumour = prob_tumour > prob_healthy
    verdict_color = "#ff4d4d" if is_tumour else "#2ecc71"
    verdict_bg = "#2a0808" if is_tumour else "#082013"
    verdict_text = "🚨 METASTASIS DETECTED" if is_tumour else "✅ NORMAL TISSUE"
    verdict_desc = "Active tumour cell clustering identified in the patch center." if is_tumour else "No significant metastatic clustering detected."

    html_output = f"""
    <div style="font-family: 'Inter', sans-serif; padding: 10px;">
        <!-- Verdict Banner -->
        <div style="background-color: {verdict_bg}; border: 2px solid {verdict_color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; box-shadow: 0 0 15px {verdict_color}33;">
            <span style="color: {verdict_color}; font-size: 26px; font-weight: 800; display: block; letter-spacing: 1px; margin-bottom: 5px;">{verdict_text}</span>
            <span style="color: #94a3b8; font-size: 14px;">{verdict_desc}</span>
        </div>
        
        <!-- Diagnostic Probability Meters -->
        <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px;">
            <h4 style="color: #f3f4f6; margin: 0 0 15px 0; font-size: 16px; font-weight: 600;">Diagnostic Analysis</h4>
            
            <!-- Tumour Probability Bar -->
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; font-weight: 500;">
                    <span style="color: #ef4444;">Tumour Probability</span>
                    <span style="color: #ef4444; font-weight: bold;">{prob_tumour * 100:.2f}%</span>
                </div>
                <div style="background-color: #1f2937; border-radius: 6px; height: 10px; overflow: hidden; width: 100%;">
                    <div style="background: linear-gradient(90deg, #c084fc, #ef4444); height: 100%; width: {prob_tumour * 100}%; border-radius: 6px; box-shadow: 0 0 8px #ef444488;"></div>
                </div>
            </div>
            
            <!-- Healthy Probability Bar -->
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; font-weight: 500;">
                    <span style="color: #10b981;">Healthy Probability</span>
                    <span style="color: #10b981; font-weight: bold;">{prob_healthy * 100:.2f}%</span>
                </div>
                <div style="background-color: #1f2937; border-radius: 6px; height: 10px; overflow: hidden; width: 100%;">
                    <div style="background: linear-gradient(90deg, #6366f1, #10b981); height: 100%; width: {prob_healthy * 100}%; border-radius: 6px; box-shadow: 0 0 8px #10b98188;"></div>
                </div>
            </div>
        </div>
    </div>
    """
    return html_output, normalized_np

def build_app(model_path, mock_mode, test_x, test_y):
    # Load model
    model = load_app_model(model_path)
    
    # Generate examples
    examples_list = create_example_images(mock_mode, test_x, test_y)
    
    # Custom Slate/Neon Theme setup
    custom_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="indigo",
        neutral_hue="slate"
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
        block_border_color="*neutral_800",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500"
    )

    # Construct the Gradio dashboard interface
    with gr.Blocks(theme=custom_theme, title="RESNET HISTOPATHOLOGY") as demo:
        # Dashboard Header
        gr.HTML("""
        <div style="text-align: center; padding: 25px 0 15px 0; border-bottom: 1px solid #1f2937; margin-bottom: 25px;">
            <h1 style="color: #c084fc; font-size: 32px; font-weight: 800; margin: 0; letter-spacing: 0.5px;">🔬 CAD-Pathology Assistant</h1>
            <p style="color: #94a3b8; font-size: 16px; margin: 6px 0 0 0;">Interactive Deep-Learning Screening for Metastatic Lymph Nodes</p>
        </div>
        """)
        
        with gr.Row(equal_height=True):
            # Column 1: Input Control Panel
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Step 1: Upload Tissue Patch")
                img_upload = gr.Image(label="Drag & Drop H&E slide image here", type="numpy", height=300)
                
                # Simple Settings
                stain_normalization_toggle = gr.Checkbox(label="Apply Macenko Stain Normalization", value=True)
                
                # Action Buttons
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Run Diagnostic", variant="primary")

            # Column 2: Dashboard Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Step 2: Diagnostic Dashboard")
                
                # Custom HTML display for verdict and probability meters
                verdict_panel = gr.HTML("<div style='text-align: center; color: #64748b; padding: 40px 0;'>Upload a slide patch and click 'Run Diagnostic' to begin.</div>")
                
                # Stain Normalized image preview
                img_normalized_preview = gr.Image(label="Normalized Image Preview", interactive=False, height=200)

        # Clickable Examples Row
        gr.Markdown("### 💡 Click any slide to try immediately:")
        gr.Examples(
            examples=examples_list,
            inputs=[img_upload, stain_normalization_toggle],
            outputs=[verdict_panel, img_normalized_preview],
            fn=lambda img, norm: run_diagnostics(img, norm, model),
            cache_examples=True
        )
        
        # Event mappings
        submit_btn.click(
            fn=lambda img, norm: run_diagnostics(img, norm, model),
            inputs=[img_upload, stain_normalization_toggle],
            outputs=[verdict_panel, img_normalized_preview]
        )
        
        # Reset button map
        def clear_inputs():
            return None, True, "<div style='text-align: center; color: #64748b; padding: 40px 0;'>Upload a slide patch and click 'Run Diagnostic' to begin.</div>", None
            
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[img_upload, stain_normalization_toggle, verdict_panel, img_normalized_preview]
        )

        # Medical Disclaimer Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding-top: 15px; border-top: 1px solid #1f2937; color: #4b5563; font-size: 12px; font-weight: 500;">
            ⚠️ <strong>Medical Disclaimer:</strong> This application is a research prototype for screening simulation. It is not an FDA-cleared diagnostic device.
        </div>
        """)
        
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pathology Assistant Gradio Application")
    parser.add_argument("--model_path", type=str, default="checkpoints/resnet18_tumor_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--mock", action="store_true", help="Run Gradio application in mock mode")
    parser.add_argument("--test_x", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_x.h5", help="Path to real test x file")
    parser.add_argument("--test_y", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_y.h5", help="Path to real test y file")
    parser.add_argument("--share", action="store_true", help="Launch Gradio app with sharing enabled")
    parser.add_argument("--port", type=int, default=7860, help="Local port to run the web server on")
    
    args = parser.parse_args()
    
    demo = build_app(
        model_path=args.model_path,
        mock_mode=args.mock,
        test_x=args.test_x,
        test_y=args.test_y
    )
    
    demo.launch(server_port=args.port, share=args.share)
