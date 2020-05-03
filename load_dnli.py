import jsonlines
from collections import Counter
from pprint import pprint


def print_dtypes(kind):
    data = jsonlines.Reader(open('./dialogue_nli/dialogue_nli_%s.jsonl' % kind, 'r')).read()
    dtype_counts = Counter([(d['dtype'], d['label']) for d in data])
    print('--- %s (%d) ---' % (kind, len(data)))
    pprint(dict(dtype_counts))
    print('')
    return data
def mapvalue(x):
    if x =="negative":
        return 2
    elif x=="positive":
        return 0
    else:
        return 1
if __name__ == '__main__':
    sen1 =[]
    sen2 =[]
    train_label=[]
    train_set =[]
    test_set=[]
    
    train_data = print_dtypes('train')
    for each in train_data:
        for key,value in each.items():
                if key == "label":
                    value1 = mapvalue(value)
                    train_label.append(value1)
                elif key =="sentence1":
                    sen1.append(value)
                elif key =="sentence2":
                    sen2.append(value)
                    #print(len(value))
                
    for idx in range(len(sen1)):  
   # for idx in range(32):
        if len(sen1[idx]) <= 32 and len(sen2[idx]) <= 32:     
            train_set.append([sen1[idx],sen2[idx], train_label[idx]])
 
#test_data = excel_data_df_test[["#1 String", "#2 String", "Quality"]] 
#    dev_data = print_dtypes('dev')
    test_data = print_dtypes('test')
    test_sen1 =[]
    test_sen2 =[]
    test_label=[]
    for each in test_data:
        for key,value in each.items():
            if key == "label":
                value1 = mapvalue(value)
                test_label.append(value1)
            elif key =="sentence1":
                test_sen1.append(value)
            elif key =="sentence2":
                test_sen2.append(value)
                
    for idx in range(len(test_sen1)):
   # for idx in range(32):  
       if len(test_sen1[idx]) <= 32 and len(test_sen2[idx]) <= 32:               
            test_set.append([test_sen1[idx],test_sen2[idx], test_label[idx]])
#    print_dtypes('verified_test')
