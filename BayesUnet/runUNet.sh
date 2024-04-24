#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J Prepare

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=10

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=8GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:30:00 

## Nazwa grantu do rozliczenia zużycia zasobów
##SBATCH -A plggraphcnn-gpu-a100
#SBATCH -A plgplgonwelocont-gpu-a100

## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu

## Plik ze standardowym wyjściem
##SBATCH --output="output_pred1.out"

## Plik ze standardowym wyjściem błędów
##SBATCH --error="error_pred1.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module add monai/1.2.0
#pip3 install nibabel
#pip3 install batchgenerators
#pip3 install torchsummary

#python3 run_preprocessing.py --config $1

#echo "Changes were saved, random control number: 934072809"
#echo "Parameters: "
#echo $1
#echo $2
#echo $3

#python3  run_training.py --config configs/$1/DROPOUT$2/config_train_$2_$3.json

#python3 run_predict.py --config configs/$1/DROPOUT$2/config_pred_Ts_$2_$3.json
python3 run_predict.py --config $1 



