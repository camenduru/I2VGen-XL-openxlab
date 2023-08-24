import os
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-video-to-video/resolve/main/open_clip_pytorch_model.bin -d /home/cache/modelscope/hub/damo/Video-to-Video -o open_clip_pytorch_model.bin")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-video-to-video/resolve/main/non_ema_0035000.pth -d /home/cache/modelscope/hub/damo/Video-to-Video -o non_ema_0035000.pth")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-video-to-video/resolve/main/v2-1_512-ema-pruned.ckpt -d /home/cache/modelscope/hub/damo/Video-to-Video -o v2-1_512-ema-pruned.ckpt")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-image-to-video/resolve/main/base_03_alldata_fps_v_80g_0789000.pth -d /home/cache/modelscope/hub/damo/Image-to-Video -o base_03_alldata_fps_v_80g_0789000.pth")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-image-to-video/resolve/main/open_clip_pytorch_model.bin -d /home/cache/modelscope/hub/damo/Image-to-Video -o open_clip_pytorch_model.bin")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/damo-image-to-video/resolve/main/v2-1_512-ema-pruned.ckpt -d /home/cache/modelscope/hub/damo/Image-to-Video -o v2-1_512-ema-pruned.ckpt")

import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0', device='cuda:0')
video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0', device='cuda:0')

def upload_file(file):
    return file.name

def image_to_video(image_in):
    if image_in is None:
        raise gr.Error('Please upload a picture!')
    print(image_in)
    output_video_path = image_to_video_pipe(image_in, output_video='./i2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path


def video_to_video(video_in, text_in):
    if video_in is None:
        raise gr.Error('Please complete the first step first.')
    if text_in is None:
        raise gr.Error('Please enter a text description.')
    p_input = {
            'video_path': video_in,
            'text': text_in
        }
    output_video_path = video_to_video_pipe(p_input, output_video='./v2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path

    with gr.Blocks() as demo:
        gr.Markdown(
        """<center><font size=7>I2VGen-XL Demo</center>
        <center><font size=3>I2VGen-XL can generate videos that closely resemble the desired content based on user-input static images and text. The generated videos are characterized by high definition (1280 * 720), widescreen (16:9), temporal coherence, and good visual quality.</center>"""
        )
    with gr.Box():
        gr.Markdown(
        """<left><font size=3>Step 1: Select the appropriate image for upload, then click 'Generate Video.' Once you are satisfied with the video, proceed to the next step.</left>"""
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image input", type="filepath", interactive=False, elem_id="image-in", height=300)
                with gr.Row():
                    upload_image = gr.UploadButton("Upload Image", file_types=["image"], file_count="single")
                    image_submit = gr.Button("Generate Video 🎬")
            with gr.Column():
                video_out_1 = gr.Video(label='The generated video.', elem_id='video-out_1', interactive=False, height=300)
    with gr.Box():
        gr.Markdown(
        """<left><font size=3>Step 2: Provide an additional English textual description for the video content, then click 'Generate High-Resolution Video.' Video generation will take approximately 2 minutes.</left>"""
        )
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Text Description", lines=2, elem_id="text-in")
                video_submit = gr.Button("Generate High-Resolution Video 🎥")
            with gr.Column():
                video_out_2 = gr.Video(label='The generated video.', elem_id='video-out_2', height=300)
    gr.Markdown("<left><font size=2>Note: If the generated video cannot be played, please try upgrading your browser or use Google Chrome browser. </left>")


    upload_image.upload(upload_file, upload_image, image_in)
    image_submit.click(fn=image_to_video, inputs=[image_in], outputs=[video_out_1])
    video_submit.click(fn=video_to_video, inputs=[video_out_1, text_in], outputs=[video_out_2])

demo.queue(max_size=10).launch()
