import glob

slurm_outs = glob.glob("slurm-*.out")
print(slurm_outs)

ok_files = []

for out_name in slurm_outs:
    print(out_name)
    f = open(out_name, "r")
    data = f.readlines() 

################## TRAINING ##################
    error = -1
    dataset = ""
    dropout = -1
    iteration = -1

    for i in range(len(data)):
        if "Parameters" in data[i]:
            dataset = data[i+1][:-1]
            dropout = int(data[i+2])
            iteration = int(data[i+3])

        if i < len(data)-2 and i != 38 and "error" in data[i].lower():
            error = i

    
    if len(data) < 1000:
    # if "The training took" not in data[-1]:
      print(dataset, "\t d:", str(dropout), "\t i:", str(iteration), "\t file:", out_name, "\t TOO SHORT!!!")

    elif error == -1:
        print(dataset, "\t d:", str(dropout), "\t i:", str(iteration), "\t file: ", out_name, "\t ok")
        ok_files.append([dataset, dropout, iteration])
    else:
        print(dataset, "\t d:", str(dropout), "\t i:", str(iteration), "\t file:", out_name, "\t EXCEPTION!!!, error:", error)


# for dataset in ["KNEE", "SKB"]:
#   for d in [2, 4, 6, 8]:
for dataset, d in [["LUNG", 4], ["LUNG", 6], ["LUNG", 8], ["KNEE", 0], ["KNEE", 2], ["KNEE", 4], ["KNEE", 6], ["SKB", 8]]:
      for i in range(5):
          if [dataset, d, i] not in ok_files:
              print("NEVER OK: ", dataset, ", d " + str(d) + ", l " + str(i))





# ################# PREDICTIONS ##################
#     error = -1
#     end = -1
#     dropout = -1
#     iteration = -1
#     eval_ok = False
#     categorize_ok = False
#     save_type_ok = False

#     known_params = False

#     for i in range(len(data)):
#         if "Parameters" in data[i]:
#             # print(out_name, data[i+1], data[i+2])
#             dropout = int(data[i+2])
#             iteration = int(data[i+3])
#             known_params = True

#         if "error" in data[i].lower():
#             error = i
        
#         if "The predictions took" in data[i]:
#             end = i

#         if known_params:
#           if (dropout == 0 and "Evaluation mode" in data[i]) or (dropout > 0 and "Training mode (Bayesian)" in data[i]):
#               eval_ok = True

#           if (dropout == 0 and "Categorize traditionally" in data[i]) or (dropout > 0 and "Predict uncertainty" in data[i]):
#               categorize_ok = True

#           if (dropout == 0 and "Saving type:  <class 'numpy.int64'>" in data[i]) or (dropout > 0 and "Saving type:  <class 'numpy.float64'>" in data[i]):
#               save_type_ok = True



#     if end > 0 and error == -1 and eval_ok and categorize_ok and save_type_ok:
#         print("d:", str(dropout), ", i:", str(iteration), ", file: ", out_name, ", ok")
#         ok_files.append([dropout, iteration])
#     else:
#         print("! d:", str(dropout), ", i:", str(iteration), ", file:", out_name, ", EXCEPTION!!!, end:", end, ", error:", error, ", eval_ok:", eval_ok, ", categorize_ok:", categorize_ok, ", save_type_ok:", save_type_ok)


# for d in [2, 4, 6, 8]:
#     for i in range(10):
#         if [d, i] not in ok_files:
#             print("NEVER OK, d " + str(d) + "i " + str(i))