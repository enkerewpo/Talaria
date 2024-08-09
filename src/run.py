from loguru import logger

logger.info("loading python modules, please wait...")

import json
import torch
import time
from TTS.api import TTS
import os
from openai import OpenAI
import base64
import threading
import re

VERSION = "0.1.0"
AUTHORS = "wheatfox <enkerewpo@hotmail.com>"

from talaria.functional import Function
from talaria import capture

logger.info("welcome to Talaria [v{}, {}]", VERSION, AUTHORS)

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_KEY = os.getenv("OPENAI_KEY")

logger.info("openai url={}, key={}", OPENAI_URL, OPENAI_KEY)

# setup GPT4
history = []

TALARIA_PROMPT_INPUT_TYPE_KEYBOARD = 0
TALARIA_PROMPT_INPUT_TYPE_FILE = 1

TALARIA_PROMPT_INPUT_TYPE_SELECT = TALARIA_PROMPT_INPUT_TYPE_KEYBOARD

# create the openai client
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_URL)

voice_last_input = ""
voice_last_input_used = False


def voice_input():
    global voice_last_input
    global voice_last_input_used
    # monitor voice_input.txt and read the content, if the
    # content changed, we return the new content
    timer = time.time()
    # remove the voice_input.txt file first and create an empty one
    if voice_last_input == "":
        if os.path.exists("voice_input.txt"):
            os.remove("voice_input.txt")
        with open("voice_input.txt", "w") as f:
            f.write("")
    while True:
        with open("voice_input.txt", "r") as f:
            content = f.read()
            if content != voice_last_input:
                voice_last_input = content
                timer = time.time()
                voice_last_input_used = False
            elif time.time() - timer > 2 and not voice_last_input_used:
                # if the content is not changed for 2 seconds, we return the content
                return content


def get_prompt_input():
    """the input interface for the prompt, can be keyboard or file input"""
    if TALARIA_PROMPT_INPUT_TYPE_SELECT == TALARIA_PROMPT_INPUT_TYPE_KEYBOARD:
        return input()
    elif TALARIA_PROMPT_INPUT_TYPE_SELECT == TALARIA_PROMPT_INPUT_TYPE_FILE:
        ret = voice_input()
        print(ret)
        global voice_last_input_used
        voice_last_input_used = True  # to avoid multiple reads of the same content
        return ret


class PromptConfig:
    """
    used to add auxiliary information to the prompt
    such as local image path
    """

    def __init__(self):
        self.local_image_path = ""

    def set_local_image_path(self, path):
        self.local_image_path = path

    def get_local_image_path(self):
        return self.local_image_path


def encode_image(image_path):
    """encode the image to base64 for the language model"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_llm_response(prompt, history, config=None):
    """get response from the language model"""
    if config is None:
        # the normal case where we just need the dialog interaction
        history.append({"role": "user", "content": prompt})
        result = client.chat.completions.create(messages=history, model="gpt-4o")
        response = result.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        return response
    else:
        # the case where we need to add some auxiliary information to the prompt
        if config.get_local_image_path() != "":
            # use gpt-4-vision-preview model
            image = encode_image(config.get_local_image_path())
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What’s in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            },
                        ],
                    }
                ],
            )
            response = result.choices[0].message.content
            history.append({"role": "assistant", "content": response})
            return response
        else:
            logger.error("not supported config: {}", config)


def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global tts
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    logger.info(f"tts using device: {device}")
    # add some pre-populated history from inference.en.json
    with open("inference.en.json", "r") as f:
        data = json.load(f)
        for d in data:
            history.append({"role": d["role"], "content": d["content"]})
    # function registration
    capture.init()


def tts_to_file_local(text, speaker_wav, language, file_path):
    logger.debug("generating audio for text: {}, file_path: {}", text, file_path)
    tts.tts_to_file(
        text=text, speaker_wav=speaker_wav, language=language, file_path=file_path
    )


def play_audio_monitor(total):
    # clean all the output_*.wav files
    for f in os.listdir("."):
        if f.startswith("output_") and f.endswith(".wav"):
            os.remove(f)
    current_index = 0
    while True:
        # check whether the current_index is in the wav_files list
        # output_001.wav, output_002.wav, output_003.wav
        # read wav from current directory
        wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
        if current_index < total:
            path = f"output_{current_index:03}.wav"
            if path in wav_files:
                logger.debug("playing audio file: {}", path)
                # play the audio
                # call cmd line to play the audio and wait for it to finish
                global tmp_voice_output
                # get the front of tmp_voice_output and write it to voice_output.txt
                # logger.debug("dump tmp_voice_output: {}", tmp_voice_output)
                with open("voice_output.txt", "w") as f:
                    f.write(tmp_voice_output[current_index])
                os.system(f"ffplay -nodisp -autoexit {path}")
                # remove the file
                os.remove(path)
                current_index += 1
        else:
            break


def format_sentence(sentence, language="en"):
    if language == "en":
        LIMIT = 80
        # each line can maximum have LIMIT characters
        # and we add \n each LIMIT characters
        # if the \n is in the middle of a word, we move it to the next line
        formated = ""
        line = ""
        for word in sentence.split(" "):
            if len(line) + len(word) + 1 > LIMIT:
                formated += line + "\n"
                line = ""
            line += word + " "
        formated += line
        return formated
    else:
        logger.error("unsupported language: {}", language)
        return sentence


tmp_voice_output = []


def main_loop():
    """main loop for the program"""

    logger.info("entering main loop, now waiting for input")
    while True:
        prompt = get_prompt_input()
        if prompt == "exit":
            break

        json_raw = get_llm_response(prompt, history)
        json_struct = json.loads(json_raw)
        logger.debug("response: {}", json_struct)
        response = json_struct["response"]

        triggered_function_names = []
        function_trigger_flag = json_struct["function_trigger_flag"]

        registered_functions_local = Function.new().get_all()
        for function_name in function_trigger_flag:
            if function_trigger_flag[function_name]:
                prefix = function_name.split("_")[0]
                triggered_function_names.append(prefix)
                # call the function only if it is registered
                if prefix in registered_functions_local:
                    func = registered_functions_local[prefix]
                    func()
                else:
                    logger.error("function {} is not registered", prefix)
        logger.info("triggered functions: {}", triggered_function_names)

        # if triggered capture, we need to re-ask the GPT and append the image screenshot.png
        if "capture" in triggered_function_names:
            config = PromptConfig()
            config.set_local_image_path("screenshot.png")
            response = get_llm_response(prompt, history, config)  # update response

        # split the response into sentences by . ? ! ;
        split_sentences = re.split(r"[.?!;]", response)
        # cleanup split_sentences, remove empty strings in the list
        split_sentences = [s.strip() for s in split_sentences if len(s.strip()) > 0]
        logger.debug("split sentences: {}", split_sentences)

        # one thread actively check the wav_files list, and play the audio in order
        threading.Thread(
            target=play_audio_monitor, args=(len(split_sentences),)
        ).start()

        global tmp_voice_output
        tmp_voice_output.clear()

        for i, sentence in enumerate(split_sentences):
            formated = format_sentence(sentence, "en")
            tmp_voice_output.append(formated)
            file_path = f"output_{i:03}.wav"
            tts_to_file_local(sentence, "./TTS/GLaDOS_01.wav", "en", file_path)


init()
main_loop()

logger.info("goodbye!")
exit(0)
