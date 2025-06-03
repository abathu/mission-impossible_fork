# tag_tokenize_only.py
# Author: Yujie Chen (modified tokenizer-only version)

import json
import stanza
import argparse
import os
import tqdm
import glob
import sys
sys.path.append("..")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Tokenize BabyLM dataset',
        description='Tokenize BabyLM dataset using Stanza'
    )
    parser.add_argument('path', type=argparse.FileType('r'),
                        nargs='+', help="Path to file(s)")

    # Get args
    args = parser.parse_args()

    # Init Stanza tokenizer only
    tokenizer = stanza.Pipeline(
        lang='en',
        processors='tokenize',  # Only tokenizer
        package="default_accurate",
        use_gpu=False
    )

    # Iterate over BabyLM files
    for file in args.path:

        print(f"Processing file: {file.name}")

        # Efficiently count lines for tqdm progress bar
        file.seek(0, os.SEEK_END)
        total_bytes = file.tell()
        file.seek(0)

        line_annotations = []
        print("Segmenting and tokenizing text line by line...")

        for line in tqdm.tqdm(file, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                split_doc = tokenizer(line)
            except Exception as e:
                print(f"⚠️ Tokenizer failed on line. Skipping. Error: {e}")
                continue

            for sentence in split_doc.sentences:
                sent_text = sentence.text.strip()
                if not sent_text:
                    continue  # Skip empty sentence

                # Build word annotations
                word_annotations = []
                for idx, token in enumerate(sentence.tokens):
                    wa = {
                        'id': idx + 1,  # 1-based index
                        'text': token.text,
                        'start_char': token.start_char,
                        'end_char': token.end_char
                    }
                    word_annotations.append(wa)

                sa = {
                    'sent_text': sent_text,
                    'word_annotations': word_annotations
                }

                la = {
                    'sent_annotations': [sa]
                }
                line_annotations.append(la)

        # Write annotations to file as a JSON
        print("Writing JSON outfile...")
        ext = '.json'
        json_filename = os.path.splitext(file.name)[0] + ext
        with open(json_filename, "w", encoding="utf-8") as outfile:
            json.dump(line_annotations, outfile, indent=4, ensure_ascii=False)
