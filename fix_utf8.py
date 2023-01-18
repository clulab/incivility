import argparse
import codecs
import ftfy


def fallback(error: UnicodeDecodeError) -> (str, int):
    bs = error.object[error.start: error.end]
    try:
        cs = bs.decode('Windows-1252')
    except UnicodeDecodeError:
        cs = bs.decode('CP437')
    return cs, error.end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    codecs.register_error('fallback', fallback)

    with open(args.input_path, encoding="utf-8", errors='fallback') as in_file:
        with open(args.output_path, 'w', encoding='utf-8-sig') as out_file:
            for line in in_file:
                if args.csv:
                    # replace double quote entities with doubled double quotes
                    # since ftfy will replace them with single quotes and mess
                    # up CSV parsing (have to first replace &amp; to handle
                    # extra bad cases like &amp;quot;)
                    line = line.replace('&amp;', '&')
                    line = line.replace('&quot;', '""')
                    line = line.replace('&ldquo;', '""')
                    line = line.replace('&rdquo;', '""')
                out_file.write(ftfy.fix_text(
                    line, uncurl_quotes=False, fix_entities=True))
