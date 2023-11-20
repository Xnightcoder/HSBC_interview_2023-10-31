# This is temporary script for combining sparated wiki_psgs_w100 passages into big paragraphs
import csv

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        data = [row for row in reader]
    return data

# Use the function
input_data = read_tsv('/home/yimin/DPR/downloads/data/wikipedia_split/psgs_w100.tsv')

data = []
output_item = {}
output_item_id = 1
output_item_title = None
output_item_text = ""
final_line = None

# Iterate through each line (dictionary)
for line in input_data:
    # Now 'line' is a dictionary representing a row in the TSV file
    # You can manipulate it as you wish
    final_line = line
    
    if(output_item_title == None):
        output_item_title = line['title']
        output_item_text = line['text'].replace('\n', '').replace('\r\n', '')
        output_item['id'] = output_item_id
    elif( output_item_title == line['title'] ):
        output_item_text += " " + line['text'].replace('\n', '').replace('\r\n', '')
    else:
        assert( output_item_title != line['title'] )
        output_item['id'] = output_item_id
        output_item_id += 1
        output_item['title'] = output_item_title
        output_item['text'] = output_item_text
        data.append(output_item)
        output_item = {}

        output_item_title = line['title']
        output_item_text = line['text'].replace('\n', '').replace('\r\n', '')
        output_item['id'] = output_item_id


## final line
output_item['id'] = output_item_id
output_item_id += 1
output_item['title'] = output_item_title
output_item['text'] = output_item_text
data.append(output_item)
    

# Specify the field names
fieldnames = ['id', 'text', 'title']

# Write data to a CSV file
with open('/home/yimin/DPR/downloads/data/wikipedia_split/psgs_w100_tsv_combine_to_1_big_json_output.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for row in data:
        writer.writerow(row)

csvfile.close()
