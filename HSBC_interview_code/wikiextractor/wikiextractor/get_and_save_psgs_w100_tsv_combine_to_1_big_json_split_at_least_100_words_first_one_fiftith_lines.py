




import os





input_file = open( '/home/yimin/DPR/downloads/data/wikipedia_split/psgs_w100_tsv_combine_to_1_big_json_output.csv' , 'r' )





output_file = open( '/home/yimin/DPR/downloads/data/wikipedia_split/psgs_w100_tsv_combine_to_1_big_json_output_get_the_first_one_fiftith_ie_first_64660_lines.csv' , 'w' )





line_no = 1
for line in input_file:
    output_file.write(line)
    line_no += 1
    
    if(line_no > 64660):
        break





input_file.close()
output_file.close()





