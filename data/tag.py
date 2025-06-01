# tag.py
# Author: Julie Kallini

# For importing utils
import json
import stanza
import argparse
import os
import tqdm
import glob
import pytest
import sys
sys.path.append("..")


# Collect test cases
test_all_files = sorted(glob.glob("babylm_data/babylm_*/*"))
test_original_files = [f for f in test_all_files if ".json" not in f]
test_json_files = [f for f in test_all_files if "_parsed.json" in f]
test_cases = list(zip(test_original_files, test_json_files))


@pytest.mark.parametrize("original_file, json_file", test_cases)
def test_equivalent_lines(original_file, json_file):
    # Read lines of file and remove all whitespace
    with open(original_file, encoding="utf-8") as f:
        original_data = "".join(f.readlines())
    original_data = "".join(original_data.split())

    with open(json_file, encoding="utf-8") as f:
        json_lines = json.load(f)
    json_data = ""
    for line in json_lines:
        for sent in line["sent_annotations"]:
            json_data += sent["sent_text"]
    json_data = "".join(json_data.split())

    # Test equivalence
    assert (original_data == json_data)


def __get_constituency_parse(sent, nlp):
    # Try parsing the doc
    try:
        parse_doc = nlp(sent.text)
    except:
        return None

    # Get set of constituency parse trees
    parse_trees = [str(sent.constituency) for sent in parse_doc.sentences]

    # Join parse trees and add ROOT
    constituency_parse = "(ROOT " + " ".join(parse_trees) + ")"
    return constituency_parse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using Stanza')
    parser.add_argument('path', type=argparse.FileType('r'),
                        nargs='+', help="Path to file(s)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")

    # Get args
    args = parser.parse_args()

    # Init Stanza NLP tools
    nlp1 = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma',
        package="default_accurate",   # You can switch to "default_fast" for faster processing
        use_gpu=True)

    # If constituency parse is needed, init second Stanza parser
    if args.parse:
        nlp2 = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            package="default_accurate",  # You can switch to "default_fast" for faster processing
            use_gpu=True)

    # No batch processing anymore, line-by-line avoids token limit issues
    # BATCH_SIZE = 500  -- no need now

    # Iterate over BabyLM files
    for file in args.path:

        print(f"Processing file: {file.name}")

        # Efficiently count lines for tqdm progress bar
        file.seek(0, os.SEEK_END)
        total_lines = file.tell()
        file.seek(0)

        # Iterate over lines in file and track annotations
        line_annotations = []
        print("Segmenting and parsing text line by line...")

        for line in tqdm.tqdm(file, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Tokenize line with stanza
            doc = nlp1(line)

            # Track annotations for this line
            sent_annotations = []
            for sent in doc.sentences:
                # Track annotations for each word
                word_annotations = []
                for token, word in zip(sent.tokens, sent.words):
                    wa = {
                        'id': word.id,
                        'text': word.text,
                        'lemma': word.lemma,
                        'upos': word.upos,
                        'xpos': word.xpos,
                        'feats': word.feats,
                        'start_char': token.start_char,
                        'end_char': token.end_char
                    }
                    word_annotations.append(wa)

                # Add constituency parse if needed
                if args.parse:
                    constituency_parse = __get_constituency_parse(sent, nlp2)
                    sa = {
                        'sent_text': sent.text,
                        'constituency_parse': constituency_parse,
                        'word_annotations': word_annotations,
                    }
                else:
                    sa = {
                        'sent_text': sent.text,
                        'word_annotations': word_annotations,
                    }
                sent_annotations.append(sa)

            la = {
                'sent_annotations': sent_annotations
            }
            line_annotations.append(la)

        # Write annotations to file as a JSON
        print("Writing JSON outfile...")
        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file.name)[0] + ext
        with open(json_filename, "w", encoding="utf-8") as outfile:
            json.dump(line_annotations, outfile, indent=4, ensure_ascii=False)
