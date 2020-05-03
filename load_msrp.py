import pandas

excel_data_df_train = pandas.read_excel('./MSRP/msr_paraphase_train.xlsx', sheet_name='Sheet1')

excel_data_df_test = pandas.read_excel('./MSRP/msr_paraphase_test.xlsx', sheet_name='Sheet1')

train_data = excel_data_df_train[["#1 String", "#2 String", "Quality"]]
test_data = excel_data_df_test[["#1 String", "#2 String", "Quality"]]

# print(train_label.head())
# print("------")
print(train_data.head())