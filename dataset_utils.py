

def sst5_generate_dataset_dict(filename):
    input_list = []
    label_list = []
    with open(filename) as f:
        
        for line_index, line in enumerate(f):
            line = line.strip()
            comma_index = line.index(',')
            label = int(line[:comma_index])
            input_sentence = line[comma_index+1:]
            
            if input_sentence.startswith('"') and input_sentence.endswith('"'):
                input_sentence = input_sentence.replace('"', '')

            label_list.append(label)
            input_list.append(input_sentence)
    return_dict = {
        'sentence' : input_list,
        'label' : label_list
    }

    return return_dict