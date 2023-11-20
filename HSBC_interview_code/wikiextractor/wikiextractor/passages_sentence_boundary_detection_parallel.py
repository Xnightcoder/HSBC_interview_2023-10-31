# The author uses multiprocessing to boost the speed of sentence boundary detection
import csv
import sys
import pysbd
import multiprocessing
import argparse



csv.field_size_limit(sys.maxsize)

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        data = []
        for row in reader:
            data.append( row )
    return data


def split_list(lst, sublist_size):
    return [lst[i:i+sublist_size] for i in range(0, len(lst), sublist_size)]


# Define the task function
def sbd_seg_task(article):
    
    result_segs = []
    seg = pysbd.Segmenter( language = "en" , clean = False )
    sentences_list = seg.segment( article )
    passage = ""
    i = 0
    
    assert( len( sentences_list ) > 0 )
    
    while True:
        try:
            passage += sentences_list[i] ###+ " "
        except Exception as e:
            print(f"Task encountered an exception: {e}")
        
        if len(passage)>=str_len_limit or i>=len(sentences_list)-1 :
            result_segs.append( passage.strip() )
            passage = ""
        i += 1
        if( i >= len(sentences_list) ):
            break
    return result_segs


def main(input_csv_file, output_csv_file, str_len_limit = 600, parallel_count = 192):
    # Use the function
    input_data = read_tsv(input_csv_file)

    input_data_split_result = split_list( input_data , parallel_count )
    data = []
    id = 1

    for group in input_data_split_result:
        # Create a multiprocessing Pool with the desired number of processes
        with multiprocessing.Pool(len(group)) as pool:
            # Apply the task function to each task in parallel
            results = pool.map(sbd_seg_task, [ e['text'] for e in group ])
        for i, result_segs in enumerate(results):
            for seg in result_segs:
                output_item = {}
                output_item['id'] = id
                output_item['text'] = seg
                output_item['title'] = group[i]['title']
                data.append( output_item )
                id += 1

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
    parser = argparse.ArgumentParser(description="Use sentence boundary detection to re-arrange passages.")
    parser.add_argument("--input_csv_file", required=True, help="The input csv file.")
    parser.add_argument("--output_csv_file", required=True, help="The output csv file.")
    parser.add_argument("--str_len_limit", required=False, default=600, help="The low bound of string length for passage")
    parser.add_argument("--parallel_count", required=False, default=192, help="The count of parallel processes to run the sentence boundary detection algorithm.")
    args = parser.parse_args()

    main(args.passage, args.question, args.str_len_limit, args.parallel_count)


