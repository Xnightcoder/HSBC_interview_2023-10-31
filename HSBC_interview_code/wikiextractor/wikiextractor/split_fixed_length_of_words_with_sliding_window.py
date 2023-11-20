import csv
import sys
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter


csv.field_size_limit(sys.maxsize)

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        data = []
        for row in reader:
            data.append( row )
    return data


def main(input_csv_file, output_csv_file, fixed_str_len = 600, sliding_window = 120):

    # Use the function
    input_data = read_tsv(input_csv_file)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = fixed_str_len,
        chunk_overlap = sliding_window,
        length_function = len,
        is_separator_regex = False,
    )
    output_item_id = 1
    data = []

    # Iterate through each line (dictionary)
    for item in input_data:
        # Now 'line' is a dictionary representing a row in the TSV file
        # You can manipulate it as you wish
        texts = text_splitter.create_documents( [ item['text'] ] )
        for segment in texts:
            output_item = {}
            output_item['id'] = output_item_id
            output_item_id += 1
            output_item['text'] = segment.page_content
            output_item['title'] = item[ 'title' ]
            data.append( output_item )

    # Specify the field names
    fieldnames = ['id', 'text', 'title']
    # Write data to a CSV file
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    csvfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a passage and a question.")
    parser.add_argument("--input_csv_file", required=True, help="The input csv file.")
    parser.add_argument("--output_csv_file", required=True, help="The output csv file.")
    parser.add_argument("--fixed_str_len", required=False, default=600, help="The string length for passage")
    parser.add_argument("--sliding_window", required=False, default=120, help="The sliding window size for the passage.")
    args = parser.parse_args()

    main(args.passage, args.question, args.fixed_str_len, args.sliding_window)


