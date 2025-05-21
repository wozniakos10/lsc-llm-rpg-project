import gradio as gr
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llm_rpg.configs import settings
from typing import AsyncGenerator
import os
import glob
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import time
import sys
import json


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
                    try:
                        result_json = json.loads(result)
                        if "text" in result_json and result_json["text"]:
                            self.current_text = result_json["text"]
                            self.recording = False  # Stop when we get final text
                    except Exception as e:
                        print(f"Error parsing recognition result: {e}")
                time.sleep(0.1)

        return (
            self.current_text,
            self.current_text,
        )  # Return text for both voice_input and msg

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
        self.template = """
        Jesteś asystentem który odpowiada za tworzenie gier rpg. Stwórz opis gry zgodnie z wymaganiem użytkownika.
        Pytanie: {question}
        """
        self.prompt = PromptTemplate.from_template(self.template)

    def get_available_models(self):
        """Get list of available models from the models directory"""
        model_files = glob.glob(f"{settings.models_path}/*.gguf")
        return [os.path.basename(model) for model in model_files]

    def create_llm(self, model_name: str):
        """Create LLM instance with selected model"""
        #  Model config
        n_gpu_layers = -1  # -1 for all layers on GPU
        n_batch = 512

        return LlamaCpp(
            model_path=f"{settings.models_path}/{model_name}",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callbacks=[self.streaming_callback],
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
            return f"Model {model_name} załadowany poprawnie!"
        except Exception as e:
            return f"Problem z załadowaniem modelu: {str(e)}"

    async def stream_chat(
        self, message: str, history: list
    ) -> AsyncGenerator[tuple[str, list], None]:
        """Stream chat messages and return streaming response"""
        if self.chain is None:
            yield "Najpierw załaduj model!", history
            return

        history = history or []

        # Create a streaming response
        response = ""
        async for chunk in self.chain.astream({"question": message}):
            response += chunk
            yield "", history + [(message, response)]

        # Final update with complete response
        history.append((message, response))
        yield "", history


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

    chatbot = gr.Chatbot(
        height=600,
        avatar_images=(
            f"{settings.assets_path}/user_avatar.png",  # User avatar
            f"{settings.assets_path}/chatbot_avatar.png",  # AI avatar
        ),
        bubble_full_width=False,
        show_copy_button=True,
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
        llm_manager.load_model, inputs=[model_dropdown], outputs=[model_status]
    )

    # Connect the chat interface
    submit.click(llm_manager.stream_chat, [msg, chatbot], [msg, chatbot])
    msg.submit(llm_manager.stream_chat, [msg, chatbot], [msg, chatbot])

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
    demo.queue().launch(share=True)
