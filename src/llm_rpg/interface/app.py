import gradio as gr
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llm_rpg.configs import settings
import os
import glob
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
import json
from json import JSONDecodeError
from llm_rpg.world_creator.dungeon_master import DungeonMaster, MessageReciever


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
                    # print(result)
                    try:
                        result_json = json.loads(result)
                        # print(result_json)
                        if "text" in result_json and result_json["text"]:
                            self.current_text = result_json["text"]
                            self.recording = False  # Stop when we get final text
                            return self.stop_recording()
                    except JSONDecodeError as e:
                        print(f"Error parsing recognition result: {e}")
                else:
                    results = recognizer.PartialResult()
                    # print(results)
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
        self.template = """
        Jesteś asystentem który odpowiada za tworzenie gier rpg. Stwórz opis gry zgodnie z wymaganiem użytkownika.
        Jezeli użytkownik nie prosi cię o stworzenie gry, to postaraj się odpowiedzieć na pytanie użytkownika.
        Pytanie: {question}
        """
        self.dm = DungeonMaster()
        self.current_dm_msg = self.dm.start()
        self.prompt = PromptTemplate.from_template(self.template)
        self.to_llm = True
        self.to_user = False

    def create_llm_chain(self, prompt_template: str):
        """Create LLM chain with custom prompt template"""
        template = PromptTemplate.from_template(prompt_template)
        return template | self.llm

    def get_available_models(self):
        """Get list of available models from the models directory"""
        model_files = glob.glob(f"{settings.models_path}/*.gguf")
        return [os.path.basename(model) for model in model_files]

    def create_llm(self, model_name: str):
        """Create LLM instance with selected model - without streaming"""
        n_gpu_layers = -1  # -1 for all layers on GPU
        n_batch = 8
        n_ctx = 8192

        return LlamaCpp(
            model_path=f"{settings.models_path}/{model_name}",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            verbose=False,
            temperature=0.7,
            max_tokens=5000,
            top_p=0.95,
            streaming=False,  # Wyłączamy streaming
        )

    def load_model(self, model_name: str):
        """Load the selected model and create chain"""
        try:
            self.llm = self.create_llm(model_name)
            self.chain = self.prompt | self.llm
            return "Model załadowany poprawnie!", model_name
        except Exception as e:
            return f"Problem z załadowaniem modelu: {str(e)}", None

    def get_initial_dm_message(self):
        """Get the initial DM message to display when model is loaded"""
        if self.llm is None:
            return []

        try:
            receiver, dm_message = self.current_dm_msg
            if receiver == MessageReciever.USER:
                # print("--"*50)
                # print("DM message for user:", dm_message)
                # print("--"*50)
                return dm_message
            return []
        except Exception as e:
            print(f"Błąd podczas pobierania wiadomości DM: {str(e)}")
            return ["Błąd inicjalizacji DM", None]

    def chat(self, message: str, history: list) -> tuple[str, list]:
        """Handle chat without streaming"""
        if self.llm is None:
            history.append((None, "Błąd,Najpierw załaduj model!"))
            return "", history

        try:
            # Checking current dm message
            receiver, dm_message = self.current_dm_msg

            if self.to_user:
                # DM sends message to user
                history.append((message, dm_message))
                # self.current_dm_msg = self.dm.refresh()
                self.to_user = False
                self.to_llm = True
                return "", history

            elif self.to_llm:
                # Sanity check, if message not empty
                if message.strip():
                    # Handling input through dm
                    receiver, (prompt_template, prompt_data) = self.dm.handle_input(
                        message
                    )
                    # print("Prompt template:", prompt_template)
                    # print("Prompt data:", prompt_data)
                    # chain with new prompt template extracted from DM
                    chain = self.create_llm_chain(prompt_template)

                    # Call llm
                    response = chain.invoke(prompt_data)
                    # print("--"*50)
                    # print("Response from LLM:", response)
                    # print("--"*50)
                    # Update DM with response
                    self.current_dm_msg = self.dm.update(response)
                    #  print(self.current_dm_msg)
                    if self.dm.check_if_refresh():
                        self.current_dm_msg = self.dm.refresh()
                        receiver, (prompt_template, prompt_data) = self.current_dm_msg
                        #  print("--"*50)
                        # print("Prompt template:", prompt_template)
                        # print("Prompt data:", response)
                        # print("--"*50)
                        chain = self.create_llm_chain(prompt_template)
                        response = chain.invoke(prompt_data)
                        # print("--"*50)
                        # print("Response after refresh:", response)
                        # print("--"*50)
                        self.current_dm_msg = self.dm.update(response)
                    # print("Current DM message:", self.current_dm_msg)

                    # Sprawdź co ma być wyświetlone
                    receiver, dm_message = self.current_dm_msg
                    self.to_user = (
                        True  # Ustaw flagę, że DM wysyła wiadomość do użytkownika
                    )
                    self.to_llm = False  # Resetuj flagę, że DM wysyła do LLM

                    if receiver == MessageReciever.USER:
                        history.append((message, dm_message))
                        return "", history
                    else:
                        return "", history

                return "", history

        except Exception as e:
            error_msg = f"Błąd: {str(e)}"
            new_history = history + [(message, error_msg)]
            return "", new_history


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
        clear_conversation_button = gr.Button(
            "Wyczyść konwersację", variant="secondary", visible=True
        )
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

    # Event handlers
    def load_model_and_init_chat(model_name):
        """Load model and return initial chat state"""
        status, current_model = llm_manager.load_model(model_name)
        initial_history = llm_manager.get_initial_dm_message()
        #  print("00"*50)
        #  print("Initial history:", initial_history)
        # print("00"*50)
        return status, current_model, [[None, initial_history]]

    load_button.click(
        load_model_and_init_chat,
        inputs=[model_dropdown],
        outputs=[model_status, current_model_button, chatbot],
    )

    clear_conversation_button.click(
        lambda: (gr.update(value=""), gr.update(value=[])),
        outputs=[msg, chatbot],
    )

    # Connect the chat interface
    submit.click(llm_manager.chat, [msg, chatbot], [msg, chatbot])
    msg.submit(llm_manager.chat, [msg, chatbot], [msg, chatbot])

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
