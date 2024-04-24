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

##python3 run_preprocessing.py --config $1

##python3 run_training_UNet.py --config $1

python3 run_predict_UNet.py --config $1

##python3 run_preprocessing.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_prep.json
##python3 modifyPlans.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_modify.json

##python3  run_training_UNet.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_$1.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_1.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_2.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_3.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_4.json

##python3 run_predict_UNet.py --config /net/people/plgrid/plgztabor/DoseModel/configs/config_pred.json

## wstawienie do kolejki: sbatch runUNet.sh
## sprawdzenie kolejki: squeue
## usunięcie z kolejki scancel Task_ID
## sprawdzenie pesymistycznego czasu startu: sbatch --test-only runUNet.sh

