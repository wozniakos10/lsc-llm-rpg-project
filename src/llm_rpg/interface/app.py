import gradio as gr
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from llm_rpg.configs import settings
from typing import Generator
import os
import glob
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
import json
from json import JSONDecodeError
from llm_rpg.world_creator.dungeon_master import DungeonMaster


class SpeechRecognizer:
    def __init__(self):
        self.q = queue.Queue()
        self.recording = False
        self.current_text = ""
        self.model = Model(lang="pl")

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.current_text = ""

        # Get default device info
        device_info = sd.query_devices(None, "input")
        samplerate = int(device_info["default_samplerate"])

        # Start recording
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            device=None,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
        ):
            recognizer = KaldiRecognizer(self.model, samplerate)

            while self.recording:
                data = self.q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    print(result)
                    try:
                        result_json = json.loads(result)
                        print(result_json)
                        if "text" in result_json and result_json["text"]:
                            self.current_text = result_json["text"]
                            self.recording = False  # Stop when we get final text
                            return self.stop_recording()
                    except JSONDecodeError as e:
                        print(f"Error parsing recognition result: {e}")
                else:
                    results = recognizer.PartialResult()
                    print(results)
                    try:
                        results_json = json.loads(results)
                        self.current_text = results_json["partial"]
                        yield self.current_text, self.current_text
                    except JSONDecodeError as e:
                        print(f"Error parsing recognition result: {e}")

    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False

        return (
            self.current_text,
            self.current_text,
        )  # Return text for both voice_input and msg


class LLMManager:
    def __init__(self):
        self.llm = None
        self.chain = None
        self.streaming_callback = StreamingStdOutCallbackHandler()
        # self.langfuse = Langfuse(
        #     public_key=settings.langfuse_public_key,
        #     secret_key=settings.langfuse_secret_key,
        #     host=settings.langfuse_host,
        # )
        # self.langfuse_callback = CallbackHandler(
        #     public_key=settings.langfuse_public_key,
        #     secret_key=settings.langfuse_secret_key,
        #     host=settings.langfuse_host,
        # )
        self.template = """
        Jesteś asystentem który odpowiada za tworzenie gier rpg. Stwórz opis gry zgodnie z wymaganiem użytkownika.
        Jezeli uytkownik nie prosi cię o stworzenie gry, to po postaraj się odpowiedzieć na pytanie użytkownika.
        Pytanie: {question}
        """
        self.dm = DungeonMaster()
        # self.dm.start()
        self.prompt = PromptTemplate.from_template(self.template)

    def create_llm_chain(self, prompt: str):
        template = PromptTemplate.from_template(prompt)
        return template | self.llm
    

    def get_available_models(self):
        """Get list of available models from the models directory"""
        model_files = glob.glob(f"{settings.models_path}/*.gguf")
        return [os.path.basename(model) for model in model_files]

    def create_llm(self, model_name: str):
        """Create LLM instance with selected model"""
        #  Model config
        n_gpu_layers = -1  # -1 for all layers on GPU
        n_batch = 8
        n_ctx = 1000

        return LlamaCpp(
            model_path=f"{settings.models_path}/{model_name}",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            callbacks=[self.streaming_callback],#, self.langfuse_callback],
            verbose=False,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            streaming=True,
        )

    def load_model(self, model_name: str):
        """Load the selected model and create chain"""
        try:
            self.llm = self.create_llm(model_name)
            self.chain = self.prompt | self.llm
            return "Model załadowany poprawnie!", model_name
        except Exception as e:
            return f"Problem z załadowaniem modelu: {str(e)}", None

    def stream_chat(
        self, message: str, history: list, dm_msg
    ) -> Generator[tuple[str, list], None, None]:
        """Stream chat messages and return streaming response"""
        if self.chain is None:
            yield "Najpierw załaduj model!", history

        if dm_msg[0].name == "USER":
            return dm_msg[1], history

        elif dm_msg[0].name == "LLM":
            handle_input = self.dm.handle_input(message)
            chain  = self.create_llm_chain(handle_input[1][0])
            response = chain.invoke(handle_input[1][1])
            dm_response = self.dm.update(response)
            self.dm.accept()
            self.dm.refresh()

            return dm_response[1], history
            



# Create instances
speech_recognizer = SpeechRecognizer()
llm_manager = LLMManager()

# Create Gradio interface
with gr.Blocks(title="LLM RPG Chat", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=llm_manager.get_available_models(),
            value=llm_manager.get_available_models()[0]
            if llm_manager.get_available_models()
            else None,
            label="Wybierz model",
            interactive=True,
        )
        load_button = gr.Button("Załaduj model", variant="primary")
        model_status = gr.Textbox(label="Status modelu", interactive=False)
        current_model_button = gr.Textbox(
            label="Aktualnie wczytany model", interactive=False
        )

    chatbot = gr.Chatbot(
        height=600,
        avatar_images=(
            f"{settings.assets_path}/user_avatar.png",  # User avatar
            f"{settings.assets_path}/chatbot_avatar.png",  # AI avatar
        ),
        bubble_full_width=False,
        show_copy_button=True,
        # type="messages", using tuples is deprecated, how to adjust it.
    )
    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Wpisz swoje pytanie...",
                container=False,
                scale=4,
            )
        with gr.Column(scale=1):
            submit = gr.Button("Wyślij", variant="primary")

    with gr.Row():
        with gr.Column(scale=4):
            voice_input = gr.Textbox(
                label="Rozpoznany tekst", interactive=False, scale=4
            )
        with gr.Column(scale=1):
            record_button = gr.Button("🎤 Nagrywaj", variant="secondary")
            stop_button = gr.Button("⏹ Stop", variant="secondary")

    # Connect the load model button
    load_button.click(
        llm_manager.load_model,
        inputs=[model_dropdown],
        outputs=[model_status, current_model_button],
    )


    dm_msg = llm_manager.dm.start()
    # Connect the chat interface
    submit.click(llm_manager.stream_chat, [msg, chatbot, dm_msg], [msg, chatbot, dm_msg])
    msg.submit(llm_manager.stream_chat, [msg, chatbot, dm_msg], [msg, chatbot, dm_msg])

    # Connect voice recording
    record_button.click(
        speech_recognizer.start_recording,
        outputs=[voice_input, msg],  # Update both voice_input and msg
    )
    stop_button.click(
        speech_recognizer.stop_recording,
        outputs=[voice_input, msg],  # Update both voice_input and msg
    )


if __name__ == "__main__":
    demo.queue().launch(share=False)
