# tag_rus.py
# Modified based on Julie Kallini
# Author Yujie Chen

import json
import stanza
import argparse
import os
import tqdm


def __get_constituency_parse(sent, nlp):
    try:
        parse_doc = nlp(sent.text)
    except:
        return None
    parse_trees = [str(sent.constituency) for sent in parse_doc.sentences]
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
    parser.add_argument('-l', '--lang', type=str, default="ru",
                        help="Language to use for tagging (default: ru)")

    args = parser.parse_args()

    # Initialize Stanza pipeline
    nlp1 = stanza.Pipeline(
        lang=args.lang,
        processors='tokenize,pos,lemma',
        package="default_accurate",
        use_gpu=True)

    if args.parse:
        try:
            nlp2 = stanza.Pipeline(
                lang=args.lang,
                processors='tokenize,pos,constituency',
                package="default_accurate",
                use_gpu=True)
        except:
            print(
                f"⚠️ Constituency parsing not available for language: {args.lang}")
            args.parse = False

    BATCH_SIZE = 5000

    for file in args.path:
        print(file.name)
        with file as f:
            lines = f.readlines()

        print("Concatenating lines...")
        lines = [l.strip() for l in lines]
        line_batches = [lines[i:i + BATCH_SIZE]
                        for i in range(0, len(lines), BATCH_SIZE)]
        text_batches = [" ".join(l) for l in line_batches]

        line_annotations = []
        print("Segmenting and parsing text batches...")
        for text in tqdm.tqdm(text_batches):
            doc = nlp1(text)

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

            la = {'sent_annotations': sent_annotations}
            line_annotations.append(la)

        print("Writing JSON outfile...")
        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file.name)[0] + ext
        with open(json_filename, "w", encoding='utf-8') as outfile:
            json.dump(line_annotations, outfile, indent=4, ensure_ascii=False)

        print("✅ Done.")
