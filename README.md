# PaliGemma Vision Language Model

A PyTorch implementation of PaliGemma for image understanding and text generation.

## Setup

### 1. Create Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model

Download PaliGemma model files from Hugging Face to your desired directory.

## Usage

### Basic Inference

```bash
python inference.py \
    --model_path="/path/to/paligemma/model" \
    --prompt="Describe this image" \
    --image_file_path="test_images/example.jpg"
```

### Parameters

- `model_path`: Path to model directory
- `prompt`: Text prompt for the model
- `image_file_path`: Path to input image
- `max_tokens_to_generate`: Maximum output tokens (default: 100)
- `temperature`: Sampling temperature (default: 0.8)
- `top_p`: Top-p sampling (default: 0.9)
- `do_sample`: Enable sampling (default: False)
- `only_cpu`: Force CPU usage (default: False)

### Example with Options

```bash
python inference.py \
    --model_path="/path/to/model" \
    --prompt="What do you see?" \
    --image_file_path="image.jpg" \
    --max_tokens_to_generate=150 \
    --do_sample=True \
    --temperature=0.7
```

## Project Structure

- `inference.py` - Main inference script
- `modeling_gemma.py` - Gemma language model
- `modeling_sliglip.py` - SigLIP vision encoder
- `processing_paligemma.py` - Image and text processing
- `utils.py` - Model loading utilities

## Requirements

- Python 3.8+
- PyTorch 2.3.0+
- 8GB+ RAM recommended
- GPU optional but recommended

## Supported Formats

- Images: JPEG, PNG, BMP, TIFF
- Automatic resizing and preprocessing