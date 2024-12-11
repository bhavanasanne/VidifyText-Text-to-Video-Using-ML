!pip install diffusers transformers accelerate moviepy imageio gtts
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import imageio
from google.colab import files
import string
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip

# Function to validate if prompt is empty
def is_empty_prompt(prompt):
    return not prompt or prompt.strip() == ''

# Function to check if prompt contains special characters
def has_special_characters(prompt):
    return any(char not in string.ascii_letters + string.digits + ' ' for char in prompt)

# Function to validate if prompt contains only letters and spaces
def has_only_letters_and_spaces(prompt):
    return all(char in string.ascii_letters + ' ' for char in prompt)

# Function to validate text prompt
def is_valid_prompt(prompt):
    if is_empty_prompt(prompt):
        return False
    if has_special_characters(prompt):
        return False
    if not has_only_letters_and_spaces(prompt):
        return False
    return True

# Function to create video from text input with TTS
def create_video_from_text_with_imageio(prompt, video_duration_seconds=5):
    try:
        # Initialize DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.to('cpu')  # Ensuring the pipeline is on CPU for offloading

        if not is_valid_prompt(prompt):
            raise ValueError("Invalid prompt. Ensure it contains only letters and spaces.")

        num_frames = video_duration_seconds * 5  # Number of frames for the video

        # Generate video frames
        result = pipe(prompt, negative_prompt="high quality, realistic, detailed", num_inference_steps=20, num_frames=num_frames)

        if not result or result.frames is None or len(result.frames) == 0:
            raise ValueError("No frames were generated. Please check the prompt and model configuration.")

        # Convert frames to a list of PIL images
        video_frames_list = []
        for frame in result.frames[0]:
            if frame is not None:
                if isinstance(frame, np.ndarray):  # Check if frame is a numpy array
                    frame = (frame * 255).astype('uint8')  # Convert frame to uint8
                    if frame.ndim == 4 and frame.shape[0] == 1:  # Handle frame shape (1, H, W, C)
                        frame = frame[0]  # Get rid of the batch dimension
                    if frame.ndim == 3 and frame.shape[2] in [3, 4]:  # Ensure frame has 3 or 4 channels
                        video_frames_list.append(frame)  # Add to list of frames
                    else:
                        raise ValueError(f"Frame has unexpected shape: {frame.shape}. Expected (H, W, C) with C=3 or 4.")
                else:
                    raise ValueError(f"Frame data is not in the expected format. Got {type(frame)}")

        if len(video_frames_list) == 0:
            raise ValueError("No video frames generated. Please check the prompt and model configuration.")

        # Create the video from frames using imageio
        video_path = '/content/temp_video.mp4'
        with imageio.get_writer(video_path, fps=5, codec='libx264') as writer:
            for frame in video_frames_list:
                writer.append_data(frame)

        # Create audio from text using gTTS
        tts = gTTS(prompt)
        audio_path = '/content/temp_audio.mp3'
        tts.save(audio_path)

        # Combine video and audio
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_video_path = '/content/final_video.mp4'
        final_clip.write_videofile(final_video_path)

        # Provide a download link for the final video
        files.download(final_video_path)

        return final_video_path

    except Exception as e:
        return f"Error: {str(e)}"

# Function to get a valid prompt
def get_valid_prompt():
    while True:
        prompt = input("Enter the text to convert into video: ")
        if is_valid_prompt(prompt):
            return prompt
        print("Invalid prompt. Please enter a valid text prompt (only letters and spaces).")

# Function to get a valid video duration
def get_valid_duration():
    while True:
        try:
            duration = int(input("Enter the video duration in seconds: "))
            if duration > 0:
                return duration
            else:
                print("Invalid duration. Please enter a positive integer.")
        except ValueError:
            print("Invalid duration. Please enter a positive integer.")

# Main execution block for TTS and video generation
if __name__ == "__main__":
    # Get a valid prompt for video generation
    prompt = get_valid_prompt()

    # Get a valid video duration from user
    video_duration_seconds = get_valid_duration()

    # Create video from text input
    video_path = create_video_from_text_with_imageio(prompt, video_duration_seconds)

    if video_path.startswith("Error:"):
        print(video_path)  # Print error message if something went wrong
    else:
        print(f"Video created successfully. You can download it from the link below.")
