# tag_rus.py
# Modified from Julie Kallini
# Author: Yujie Chen

import os
import json
import argparse
import stanza
import tqdm
from stanza.pipeline.core import DownloadMethod


def __get_constituency_parse(sent, nlp):
    try:
        parse_doc = nlp(sent.text)
    except Exception:
        return None
    parse_trees = [str(s.constituency) for s in parse_doc.sentences]
    return f"(ROOT {' '.join(parse_trees)})"


def collect_txt_files(folder):
    """Recursively collect all .txt files from the given folder."""
    file_paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".txt"):
                file_paths.append(os.path.join(root, name))
    return sorted(file_paths)


def main(args):
    # Collect files
    file_paths = collect_txt_files(args.folder)
    if not file_paths:
        raise FileNotFoundError(f"No .txt files found in: {args.folder}")
    print(f"📂 Found {len(file_paths)} text files in {args.folder}")

    # Load Stanza pipeline
    print("🚀 Loading Stanza pipeline...")
    nlp1 = stanza.Pipeline(
        lang=args.lang,
        processors='tokenize,pos,lemma',
        package="default_accurate",
        dir=os.environ.get("STANZA_RESOURCES_DIR"),
        download_method=DownloadMethod.REUSE_RESOURCES,
        use_gpu=True
    )

    # Optional constituency parser
    nlp2 = None
    if args.parse:
        try:
            nlp2 = stanza.Pipeline(
                lang=args.lang,
                processors='tokenize,pos,constituency',
                package="default_accurate",
                dir=os.environ.get("STANZA_RESOURCES_DIR"),
                download_method=DownloadMethod.REUSE_RESOURCES,
                use_gpu=True
            )
        except Exception:
            print(
                f"⚠️ Constituency parsing not available for language: {args.lang}")
            args.parse = False

    # Process each file
    BATCH_SIZE = 5000
    for path in file_paths:
        print(f"📄 Processing: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        text_batches = [
            " ".join(lines[i:i + BATCH_SIZE])
            for i in range(0, len(lines), BATCH_SIZE)
        ]

        line_annotations = []
        print("🧠 Annotating...")
        for text in tqdm.tqdm(text_batches):
            doc = nlp1(text)
            sent_annotations = []

            for sent in doc.sentences:
                word_annotations = [{
                    'id': word.id,
                    'text': word.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'feats': word.feats,
                    'start_char': token.start_char,
                    'end_char': token.end_char
                } for token, word in zip(sent.tokens, sent.words)]

                sa = {
                    'sent_text': sent.text,
                    'word_annotations': word_annotations
                }

                if args.parse:
                    sa['constituency_parse'] = __get_constituency_parse(
                        sent, nlp2)

                sent_annotations.append(sa)

            line_annotations.append({'sent_annotations': sent_annotations})

        # Save result
        output_path = os.path.splitext(
            path)[0] + ('_parsed.json' if args.parse else '.json')
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(line_annotations, out_f, indent=4, ensure_ascii=False)

        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using Stanza'
    )
    parser.add_argument('folder', type=str,
                        help="Path to folder containing .txt files")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")
    parser.add_argument('-l', '--lang', type=str, default="ru",
                        help="Language to use for tagging (default: ru)")

    args = parser.parse_args()
    main(args)
