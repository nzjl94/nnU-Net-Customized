dataset = "KNEE"

dropout_dict = {'00': 0.0,'02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8}

for dropout in dropout_dict.keys():
  ############## TRAIN ########################
  print("------------------ TRAIN ------------------")
  for fold in range(5):
    f = open(dataset + "/DROPOUT" + dropout + "/config_train_" + dropout + "_" + str(fold) + ".json", "r")
    data = f.readlines()
    continue_training = -1
    right_fold = False
    # right_plans = False
    # right_prep = False
    right_out = False
    right_log = False
    right_net = False
    right_checkpoint = False
    right_dropout = False

    for line in data:
      for other_dropout in [el for el in dropout_dict.keys() if el != dropout and el != "00"]:
        if other_dropout in line:
          print("EXCEPTION")

      for other_dropout_val in [el for el in dropout_dict.values() if el != dropout_dict[dropout] and el != 0.0]:
        if str(other_dropout_val) in line:
          print("EXCEPTION")
      
    if "\"continue_training\": false" in data[19]:
      continue_training = 0
    if "\"continue_training\": true" in data[19]:
      continue_training = 1

    if "\"fold\": " + str(fold) in data[1]:
      right_fold = True
    # if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetPlansv2.1_plans_2D.pkl" in data[3]:
    #   right_plans = True
    # if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetData_plans_v2.1_2D_stage0" in data[4]:
    #   right_prep = True
    if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/DROPOUT" + dropout + "/MODELS/" in data[7]:
      right_out = True
    if "log" + str(fold) + ".txt" in data[8]:
      right_log = True
    # if "\"2d\"" in data[10]:
    #   right_net = True
    if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/DROPOUT" + str(dropout) + "/MODELS/fold_" + str(fold) + "_model_best.model" in data[20]:
      right_checkpoint = True
    if str(dropout_dict[dropout]) in data[22]:
      right_dropout = True
      
    f.close()
    if right_fold  and right_out and right_log and right_checkpoint: # and right_plans and right_prep and right_net
      print("ok", continue_training)
    else:
      print("EXCEPTION train ", dropout, fold)



  ############## PREDICT TS ###########################
  print("------------------ PREDICT ------------------")
  if dropout == '00':
    ran = 1
  else:
    ran = 10

  for iteration in range(ran):
    right_net = False
    right_in = False
    right_out = False
    right_plans = False
    right_checkpoints = 0
    right_dropout = False

    f = open(dataset + "/DROPOUT" + dropout + "/config_pred_Ts_" + dropout + "_" + str(iteration) + ".json", "r")
    data = f.readlines()

    for line in data:
      for other_dropout in [el for el in dropout_dict.keys() if el != dropout]:
        if other_dropout in line:
          print("EXCEPTION")
      
      for other_dropout_val in [el for el in dropout_dict.values() if el != dropout_dict[dropout]and el != 0.0]:
        if str(other_dropout_val) in line:
          print("EXCEPTION")

    if "\"2d\"" in data[2]:
      right_net = True

    if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/imagesTs" in data[3]:
      right_in = True

    if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/DROPOUT" + str(dropout) + "/predsTs_" + str(iteration) in data[4]:
      right_out = True

    if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/preprocessed/nnUNetPlansv2.1_plans_2D.pkl" in data[5]:
      right_plans = True
    
    for fold in range(5):
      if "/net/tscratch/people/plgsliwinska/venv/" + dataset + "/DROPOUT" + str(dropout) + "/MODELS/fold_" + str(fold) + "_model_best.model" in data[8+fold]:
        right_checkpoints += 1
    
    if str(dropout_dict[dropout]) in data[15]:
      right_dropout = True
      
    f.close()
    if right_net and right_in and right_out and right_plans and right_checkpoints == 5 and right_dropout:
      print("ok")
    else:
      print("EXCEPTION pred", dropout, iteration)
  