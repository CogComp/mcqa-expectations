export EPOCHS=5
echo "Processing Aristo"
python utils/preprocess.py --path $ARISTO_DATA_PATH --dataset aristo --alignment_path ./alignment_ids
echo "Processing QASC"
python utils/preprocess.py --path $QASC_DATA_PATH --dataset qasc --alignment_path ./alignment_ids
echo "Processing RACE"
python utils/preprocess.py --path $RACE_DATA_PATH --dataset race --alignment_path ./alignment_ids
echo "Generating augmented, perturbed data for Aristo"
python utils/generate_data.py --augmented_data --perturbed_data --epoch $EPOCHS --path $ARISTO_DATA_PATH
echo "Generating augmented, perturbed data for QASC"
python utils/generate_data.py --augmented_data --perturbed_data --epoch $EPOCHS --path $QASC_DATA_PATH
echo "Generating augmented, perturbed data for RACE"
python utils/generate_data.py --augmented_data --perturbed_data --epoch $EPOCHS --path $RACE_DATA_PATH
