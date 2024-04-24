dropout_dict = {'02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8}

dataset = "LUNG_Malign"

for dropout in dropout_dict.keys():
  ################ TRAIN ################
  for fold in range(5):
    with open("LUNG/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'r', encoding='utf-8') as file: 
        data = file.readlines() 
    
    if "plans_file_path" in data[3]:
      #  data[3] = "  \"plans_file_path\": \"/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetPlansv2.1_plans_2D.pkl\",\n"
      data[3]= data[3].replace("dataSegm", dataset)
    else:
       raise("Wrong index 3")
    
    if "folder_with_preprocessed_data" in data[4]:
      #  data[4] = "  \"folder_with_preprocessed_data\": \"/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetData_plans_v2.1_2D_stage0\" ,\n"
      data[4] = data[4].replace("dataSegm", dataset)
    else:
       raise("Wrong index 4")
    
    if "output_folder" in data[7]:
       data[7] = data[7].replace("dataSegm", dataset)
    else:
       raise("Wrong index 7")
    
    if "log_file" in data[8]:
       data[8] = data[8].replace("dataSegm", dataset)
    else:
       raise("Wrong index 8")

    # if "network_type" in data[10]:
    #    data[10] = "  \"network_type\": \"2d\",\n"
    # else:
    #    raise("Wrong index 10")
      
    if "continue_training" in data[19]:
       data[19] = "  \"continue_training\": false,\n"
    else:
       raise("Wrong index 19")
    
    if "checkpoint" in data[20]:
       data[20] = data[20].replace("dataSegm", dataset)
    else:
       raise("Wrong index 20")


    with open(dataset + "/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'w', encoding='utf-8') as file: 
        file.writelines(data) 






  # ################ PRED ################
  # for iteration in range(10):
  #   with open("LUNG/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'r', encoding='utf-8') as file: 
  #       data = file.readlines() 
      

  #   # if "network_type" in data[2]:
  #   #    data[2] = "  \"network_type\": \"2d\",\n"
  #   # else:
  #   #    raise("Wrong index 2")

  #   if "input_folder" in data[3]:
  #      data[3] = data[3].replace("LUNG", dataset)
  #   else:
  #      raise("Wrong index 3")
    
  #   if "output_folder" in data[4]:
  #      data[4] = data[4].replace("LUNG", dataset)
  #   else:
  #      raise("Wrong index 4")
    
  #   # if "plans_file_path" in data[5]:
  #   #    data[5] = "  \"plans_file_path\": \"/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetPlansv2.1_plans_2D.pkl\",\n"
  #   # else:
  #   #    raise("Wrong index 5")
    

  #   for f in range(5):
  #      if "\"" + str(f+1) + "\":" in data[8+f]:
  #         data[8+f] = data[8+f].replace("LUNG", dataset)
  #      else:
  #         raise("Wrong index", 8+f)
      

  #   with open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'w', encoding='utf-8') as file: 
  #       file.writelines(data) 










  # for iteration in range(10):
  #   with open("LUNG/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'r', encoding='utf-8') as file: 
  #       data = file.readlines() 
      

  #   for change_idx in range(len(data)):
  #      data[change_idx] = data[change_idx].replace("LUNG", dataset)
      
  #   with open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'w', encoding='utf-8') as file: 
  #       file.writelines(data) 




  #   with open("LUNG/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'r', encoding='utf-8') as file: 
  #       data = file.readlines() 
      

  #   for change_idx in range(len(data)):
  #      data[change_idx] = data[change_idx].replace("dataSegm", dataset)
      
  #   with open(dataset + "/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", 'w', encoding='utf-8') as file: 
  #       file.writelines(data) 



  # for iteration in range(10):
  #   with open("LUNG/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'r', encoding='utf-8') as file: 
  #       data = file.readlines() 
      

  #   for change_idx in range(len(data)):
  #      data[change_idx] = data[change_idx].replace("LUNG", dataset)
      
  #   with open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", 'w', encoding='utf-8') as file: 
  #       file.writelines(data) 