import gradio as gr
import requests


def load_video_data(video_path):
    with open(video_path, 'rb') as file:
        video_data = file.read()
    return video_data


class ChatAgent:
    def __init__(self):
        pass

    def answer(self, video_path, prompt, max_new_tokens, num_beams, temperature):
        url = 'http://127.0.0.1:5000/video_qa'
        files = {'video': open(video_path, 'rb')}
        data = {'question': prompt, 'temperature': temperature}
        response = requests.post(url, files=files, data=data)
        if response.status_code != 200:
            return f"Something went wrong: {response.text}"
        else:
            return response.json()["answer"]


def gradio_reset():
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(placeholder='Please upload your video first', interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
    )


def upload_video(gr_video):
    if gr_video is None:
        return None, gr.update(interactive=True, placeholder='Please upload video/image first!'), gr.update(
            interactive=True)
    else:
        print(f"Get video: {gr_video}")
        return (
            gr.update(interactive=True),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False)
        )


def gradio_ask(user_message, chatbot):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot


def gradio_answer(video_path, chatbot, num_beams, temperature):
    if len(chatbot) == 0 or video_path is None:
        return chatbot

    response = agent.answer(video_path=video_path, prompt=chatbot[-1][0], max_new_tokens=200, num_beams=num_beams,
                            temperature=temperature)
    print(f"Question: {chatbot[-1][0]} Answer: {response}")
    chatbot[-1][1] = response
    return chatbot


agent = ChatAgent()


def main():
    with gr.Blocks(title="VideoHub",
                   css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
        with gr.Row():
            with gr.Column(scale=0.5, visible=True) as video_upload:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=360)

                upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam search numbers",
                )

            with gr.Column(visible=True) as input_raws:
                chatbot = gr.Chatbot(elem_id="chatbot", label='VideoHub')
                with gr.Row():
                    with gr.Column(scale=0.7):
                        text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first',
                                                interactive=False, container=False)
                    with gr.Column(scale=0.15, min_width=0):
                        run = gr.Button("💭Send")
                    with gr.Column(scale=0.15, min_width=0):
                        clear = gr.Button("🔄Clear")

        upload_button.click(upload_video, [up_video],
                            [up_video, text_input, upload_button])

        text_input.submit(gradio_ask, [text_input, chatbot],
                          [text_input, chatbot]).then(
            gradio_answer, [up_video, chatbot, num_beams, temperature], [chatbot]
        )
        run.click(gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
            gradio_answer, [up_video, chatbot, num_beams, temperature], [chatbot]
        )
        run.click(lambda: "", None, text_input)
        clear.click(gradio_reset, [],
                    [chatbot, up_video, text_input, upload_button], queue=False)
    demo.launch(server_name="0.0.0.0", server_port=7868)


if __name__ == '__main__':
    main()
