from loguru import logger

logger.info("loading python modules, please wait...")

import torch
import time
from TTS.api import TTS
import os
from openai import OpenAI
import threading

VERSION = "0.1.0"
AUTHORS = "wheatfox <enkerewpo@hotmail.com>"

logger.info("welcome to Talaria [v{}, {}]", VERSION, AUTHORS)

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_KEY = os.getenv("OPENAI_KEY")

logger.info("openai url={}, key={}", OPENAI_URL, OPENAI_KEY)

# setup GPT4
history = []

TALARIA_PROMPT_INPUT_TYPE_KEYBOARD = 0
TALARIA_PROMPT_INPUT_TYPE_FILE = 1

TALARIA_PROMPT_INPUT_TYPE_SELECT = TALARIA_PROMPT_INPUT_TYPE_FILE

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


def get_llm_response(prompt, history):
    """get response from the language model"""
    history.append({"role": "user", "content": prompt})
    result = client.chat.completions.create(messages=history, model="gpt-4o")
    response = result.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    return response


def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global tts
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    logger.info(f"tts using device: {device}")
    # add some pre-populated history
    history.append(
        {
            "role": "assistant",
            "content": "Hello, I am Tara, an AI assistant.",
        },
    )
    history.append(
        {
            "role": "user",
            "content": "Remember, for every response you give me, the response should be only in one paragraph and no more than 9 sentences.",
        },
    )


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
    time.sleep(1)
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
                logger.debug("dump tmp_voice_output: {}", tmp_voice_output)
                with open("voice_output.txt", "w") as f:
                    f.write(tmp_voice_output[current_index])
                os.system(f"ffplay -nodisp -autoexit {path}")
                # remove the file
                os.remove(path)
                current_index += 1
        else:
            break


tmp_voice_output = []


def main_loop():
    """main loop for the program"""

    logger.info("entering main loop, now waiting for input")
    while True:
        prompt = get_prompt_input()
        if prompt == "exit":
            break

        print("> ", end="")
        response = get_llm_response(prompt, history)
        print(response)

        # split the response into sentences by . ? ! ;
        import re

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

        def format_sentence(sentence):
            # each line can maximum have 95 characters
            # and we add \n each 95 characters
            # if the \n is in the middle of a word, we move it to the next line
            formated = ""
            line = ""
            for word in sentence.split(" "):
                if len(line) + len(word) + 1 > 95:
                    formated += line + "\n"
                    line = ""
                line += word + " "
            formated += line
            return formated

        for i, sentence in enumerate(split_sentences):
            formated = format_sentence(sentence)
            tmp_voice_output.append(formated)
            file_path = f"output_{i:03}.wav"
            tts_to_file_local(sentence, "./GLaDOS_01.wav", "en", file_path)


init()
main_loop()

logger.info("goodbye!")
exit(0)
