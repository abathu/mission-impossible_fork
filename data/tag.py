# tag.py
# Author: Julie Kallini (modified for streaming batch-safe writing)

import sys
sys.path.append("..")

import pytest
import glob
import tqdm
import os
import argparse
import stanza
import json
import gc


test_all_files = sorted(glob.glob("babylm_data/babylm_*/*"))
test_original_files = [f for f in test_all_files if ".json" not in f]
test_json_files = [f for f in test_all_files if "_parsed.json" in f]
test_cases = list(zip(test_original_files, test_json_files))


@pytest.mark.parametrize("original_file, json_file", test_cases)
def test_equivalent_lines(original_file, json_file):
    original_data = "".join(open(original_file).readlines()).replace(" ", "").replace("\n", "")
    json_lines = json.load(open(json_file))
    json_data = "".join(sent["sent_text"] for line in json_lines for sent in line["sent_annotations"])
    json_data = "".join(json_data.split())
    assert (original_data == json_data)


def __get_constituency_parse(sent, nlp):
    try:
        parse_doc = nlp(sent.text)
        parse_trees = [str(s.constituency) for s in parse_doc.sentences]
        return "(ROOT " + " ".join(parse_trees) + ")"
    except:
        return None


if __name__ == "__main__":

    print("\n\n\nThis is the Lemma version tagging, if the normal version, modify the sa part to back  \n\n\n")

    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using Stanza')
    parser.add_argument('path', type=argparse.FileType('r'),
                        nargs='+', help="Path to file(s)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")

    args = parser.parse_args()

    # nlp1 = stanza.Pipeline(
    #     lang=  'zh-hans', #'zh',
    #     processors='tokenize,pos,lemma',
    #     package="default_accurate",
    #    dir='/home/s2678328/.cache/my_stanza/my_stanza',
    #     use_gpu=True
    # )

    nlp1 = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,lemma',
    package=None,  # Let it use what's locally available
    dir='/home/s2678328/.cache/en_stanza/en_stanza',
    use_gpu=True,
    allow_download=False  # Optional: prevent fallback to HuggingFace
)


    nlp_cpu = stanza.Pipeline(
        lang= 'en',  # 'zh',
        processors='tokenize,pos,lemma',
        package=None, #"default_accurate",
        dir='/home/s2678328/.cache/en_stanza/en_stanza',
        use_gpu=False
    )

    if args.parse:
        print("[Warning] Constituency parsing is not supported for Russian. Disabling.")
        args.parse = False

    BATCH_SIZE = 2000

    for file in args.path:
        print(f"Processing: {file.name}")
        lines = [l.strip() for l in file.readlines()]
        line_batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
        text_batches = [" ".join(batch) for batch in line_batches]

        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file.name)[0] + ext
        print(f"Writing to: {json_filename}")

        with open(json_filename, "w") as outfile:
            outfile.write("[\n")

            for i, text in enumerate(tqdm.tqdm(text_batches)):
                try:
                    doc = nlp1(text)
                except RuntimeError as e:
                    if 'CUDNN_STATUS_NOT_SUPPORTED' in str(e):
                        print(f"[Warning] CuDNN crash on batch {i} – retrying on CPU.")
                        doc = nlp_cpu(text)
                    else:
                        raise e

                sent_annotations = []

                for sent in doc.sentences:
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

                    sa = {
                        # 'sent_text': sent.text,
                        'sent_text':" ".join([word.lemma for word in sent.words if word.lemma is not None]),

                        'word_annotations': word_annotations
                    }

                    if args.parse:
                        sa['constituency_parse'] = __get_constituency_parse(sent, nlp_cpu)

                    sent_annotations.append(sa)

                la = {'sent_annotations': sent_annotations}
                json.dump(la, outfile, indent=4)

                if i < len(text_batches) - 1:
                    outfile.write(",\n")
                else:
                    outfile.write("\n")

                del doc, sent_annotations, la
                gc.collect()

            outfile.write("]\n")

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         prog='Tag BabyLM dataset',
#         description='Tag BabyLM dataset using Stanza')
#     parser.add_argument('path', type=argparse.FileType('r'),
#                         nargs='+', help="Path to file(s)")
#     parser.add_argument('-p', '--parse', action='store_true',
#                         help="Include constituency parse")

#     args = parser.parse_args()

#     nlp1 = stanza.Pipeline(
#         lang='ru',
#         processors='tokenize,pos,lemma',
#         package="default_accurate",
#         dir='/home/s2678328/.cache/my_stanza',
#         use_gpu=True
#     )

#     nlp2 = None
#     if args.parse:
#         print("[Warning] Constituency parsing is not supported for Russian. Disabling.")
#         args.parse = False

#     BATCH_SIZE = 2000

#     for file in args.path:
#         print(f"Processing: {file.name}")
#         lines = [l.strip() for l in file.readlines()]
#         line_batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
#         text_batches = [" ".join(batch) for batch in line_batches]

#         ext = '_parsed.json' if args.parse else '.json'
#         json_filename = os.path.splitext(file.name)[0] + ext
#         print(f"Writing to: {json_filename}")

#         with open(json_filename, "w") as outfile:
#             outfile.write("[\n")

#             for i, text in enumerate(tqdm.tqdm(text_batches)):
#                 doc = nlp1(text)
#                 sent_annotations = []

#                 for sent in doc.sentences:
#                     word_annotations = []
#                     for token, word in zip(sent.tokens, sent.words):
#                         wa = {
#                             'id': word.id,
#                             'text': word.text,
#                             'lemma': word.lemma,
#                             'upos': word.upos,
#                             'xpos': word.xpos,
#                             'feats': word.feats,
#                             'start_char': token.start_char,
#                             'end_char': token.end_char
#                         }
#                         word_annotations.append(wa)

#                     sa = {
#                         'sent_text': sent.text,
#                         'word_annotations': word_annotations
#                     }

#                     if args.parse:
#                         sa['constituency_parse'] = __get_constituency_parse(sent, nlp2)

#                     sent_annotations.append(sa)

#                 # write each line annotation (1 per batch)
#                 la = {'sent_annotations': sent_annotations}
#                 json.dump(la, outfile, indent=4)

#                 if i < len(text_batches) - 1:
#                     outfile.write(",\n")
#                 else:
#                     outfile.write("\n")

#                 # Clean memory after each batch
#                 del doc, sent_annotations, la
#                 gc.collect()

#             outfile.write("]\n")