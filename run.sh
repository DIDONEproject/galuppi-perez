#!/bin/env sh

LOG_FILE='experiments.log'
echo "Starting at $(date)"

####################################
####### BACKUP EXPERIMENTS #########
####################################

# if [[ -d "experiments.back.1" ]]
# then
#   last_index=$(ls experiments.back.* -d | awk -F '.' '/experiments./{print $3}' | sort -n | tail -1)
# else
#   last_index=0
# fi
# if [[ -d "experiments-holdout_?.?" ]]
# then
#   mv experiments-holdout_?.? experiments.back.$((last_index+1))
# fi
####################################

##########################################
######## ECTRACTING FEATURES #############
##########################################

# echo
# echo "Extracting features" | tee -a $LOG_FILE
# ./dust -m modeling features | tee -a $LOG_FILE
# echo

####################################
######### EXPERIMENTS ##############
####################################

echo
echo "Starting Experiment with Full AutoML" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling blackbox automl --skipsearches --skipbagfitting >> $LOG_FILE 2>&1

echo
echo "Starting Experiment with Linear AutoML" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling linear automl --skipsearches --skipbagfitting >> $LOG_FILE 2>&1

echo
echo "Starting Experiment with Tree AutoML" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling tree automl --skipsearches --skipbagfitting >> $LOG_FILE 2>&1

# permutation plot of automl
# echo
# echo "Starting Permutation-based Feature Analysis" | tee -a $LOG_FILE
# echo "------------------" >> $LOG_FILE
# echo "------------------" >> $LOG_FILE
# echo "------------------" >> $LOG_FILE
# ./dust -m modeling blackbox permutation >> $LOG_FILE 2>&1

# baseline
echo
echo "Starting Baseline grid-search-based methods" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
# ./dust -m modeling baseline gridsearch >> $LOG_FILE 2>&1

# showing model structure
echo
echo "Showing Model Structures" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling.show_models | tee -a $LOG_FILE

# perform post-hoc analysis
echo
echo "Performing Inspection!" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling inspection | tee -a $LOG_FILE

# perform post-hoc analysis
echo
echo "Performing Post-hoc analysis!" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling post_analysis | tee -a $LOG_FILE

####################################
########## FINAL TEST ##############
####################################

echo
echo "Final Test on Suspected Arias!" | tee -a $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
echo "------------------" >> $LOG_FILE
./dust -m modeling test | tee -a $LOG_FILE

echo
