import argparse
import codecs


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
    args = parser.parse_args()

    codecs.register_error('fallback', fallback)

    with open(args.input_path, encoding="utf-8", errors='fallback') as in_file:
        with open(args.output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(in_file.read())
