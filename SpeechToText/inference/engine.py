from typing import Dict, List, Any
import sys, os, re
from tqdm import tqdm

import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.utils import preprocess_batch, postprocess_batch
from IndicTransTokenizer.tokenizer import IndicTransTokenizer


class EndpointHandler():
    def __init__(self, direction = "en-indic", quantization = ""):
        self.model_name = "ai4bharat/indictrans2-en-indic-1B"

        self.utterance_pattern = re.compile(r"^\d+$")
        self.timestamp_pattern = re.compile(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)")

        self.BATCH_SIZE = 16
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None

        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        self.tokenizer = IndicTransTokenizer(direction=direction)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig
        )

        if qconfig==None:
            self.model = self.model.to(self.DEVICE)
            self.model.half()

        self.model.eval()


    def batch_translate(self, input_sentences, src_lang, tgt_lang):
        translations = []
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i : i + self.BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch, entity_map = preprocess_batch(
                batch, src_lang=src_lang, tgt_lang=tgt_lang
            )

            # Tokenize the batch and generate input encodings
            inputs = self.tokenizer(
                batch,
                src=True,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text
            generated_tokens = self.tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(), src=False
            )

            # Postprocess the translations, including entity replacement
            translations += postprocess_batch(
                generated_tokens, lang=tgt_lang, placeholder_entity_map=entity_map
            )

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return translations


    def read_srt(self, srt_path):
        data = []
        with open(srt_path, 'r', encoding='utf-8') as fp:
            utterance_ind = ""
            start_end = ""
            text = ""
            for ind, line in enumerate(fp.readlines()):
                line = line.strip()
                if re.search(self.utterance_pattern, line) is not None:
                    utterance_ind = line
                elif re.search(self.timestamp_pattern, line) is not None:
                    start_end = line
                else:
                    text += line

                if utterance_ind!='' and start_end!='' and text!='':
                    data.append({'utterance_ind': utterance_ind, 'start_end': start_end, 'text': text})
                    utterance_ind = ''
                    start_end = ''
                    text = ''

        return data

    def test(self, inputs) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: (transcript_path : 'str', src_lang : 'str', tgt_lang : 'str')
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """

        src_lang = inputs["src_lang"]
        tgt_lang = inputs["tgt_lang"]
        transcript_path = inputs["transcript_path"]

        output_translations = []
        if self.model is not None:
            transcriptions = self.read_srt(transcript_path)
            trans_sents = [entry['text'] for entry in transcriptions]
            indic_translations = self.batch_translate(trans_sents, src_lang, tgt_lang)

            for i in tqdm(range(len(transcriptions))):
                entry = transcriptions[i]
                entry['text'] = indic_translations[i]
                output_translations.append(entry)

            return output_translations
        else:
            return []

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: (transcript_path : 'str', src_lang : 'str', tgt_lang : 'str')
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """

        inputs = data.pop("inputs",data)

        src_lang = inputs["src_lang"]
        tgt_lang = inputs["tgt_lang"]
        transcript_path = inputs["transcript_path"]

        output_translations = []
        if self.model is not None:
            transcriptions = self.read_srt(transcript_path)
            trans_sents = [entry['text'] for entry in transcriptions]
            indic_translations = self.batch_translate(trans_sents, src_lang, tgt_lang)

            for i in tqdm(range(len(transcriptions))):
                entry = transcriptions[i]
                entry['text'] = indic_translations[i]
                output_translations.append(entry)

            return output_translations
        else:
            return []
