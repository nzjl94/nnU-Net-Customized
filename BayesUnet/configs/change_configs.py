desired_continue = "true"
dataset = "LUNG_Malign"

dropout_dict = {'00': 0.0, '02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8}

data  = []
for dropout in dropout_dict.keys():
  for fold in range(5):
    # with open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iter) + ".json", 'r', encoding='utf-8') as file:
    with open(dataset + "/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'r', encoding='utf-8') as file: 
        data = file.readlines() 
      
    # change_idx = 19
    # if "\"continue_training\":" in data[change_idx]:
    #     data[change_idx] = "  \"continue_training\": " + desired_continue + ",\n"
    # else: 
    #     print("EXCEPTION")

    change_idx = 15
    if "numOfEpochs" in data[change_idx]:
       data[change_idx] = "  \"numOfEpochs\": 50000,\n"
    else:
       print("Wrong index")


    # for change_idx in range(len(data)):
    #    data[change_idx] = data[change_idx].replace("dataSegm", "LUNG")
       
    # with open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iter) + ".json", 'w', encoding='utf-8') as file: 
    with open(dataset + "/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'w', encoding='utf-8') as file: 
        file.writelines(data) 
      
